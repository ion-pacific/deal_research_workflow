-- 006: org_portfolio_status gains `via_counterparty_manager_portfolio`
-- (Todd "do we hold X" look-through via fund managers we hold deals with).
-- Additive: in_portfolio and all existing keys unchanged, so the dce
-- one-pager callers are unaffected. Body otherwise = live definition as
-- of 2026-11-03 (it had drifted from 002_todd.sql).

CREATE OR REPLACE FUNCTION dealcloud.org_portfolio_status(org_ids integer[])
 RETURNS jsonb
 LANGUAGE sql
 STABLE
AS $function$
WITH
counterparty_all AS (
    SELECT d.id AS deal_id, d.name, d.transaction_type, d.status,
           d.new_deal_date AS date
      FROM dealcloud.deal d
     WHERE d.organization_id = ANY(org_ids)
       AND d.status IN ('Portfolio Company', 'Partnership')
),
counterparty_top AS (
    SELECT * FROM counterparty_all
     ORDER BY date DESC NULLS LAST, deal_id DESC
     LIMIT 5
),
underlying_all AS (
    -- Include both is_underlying and is_value_driver. DealCloud's two
    -- source fields (Organization.RelatedDeal -> is_underlying,
    -- Deal.Companiesexposure -> is_value_driver) aren't perfectly
    -- consistent: ~100 DUC rows have only is_value_driver=TRUE. The
    -- design intent is that every value driver is also an underlying
    -- company. So treat any DUC row as a portfolio-relationship signal.
    SELECT d.id AS deal_id, d.name AS deal_name,
           d.organization_id AS parent_org_id,
           o_parent.name      AS parent_org_name,
           d.status, d.new_deal_date AS date,
           duc.connection_source,
           duc.derived_n_docs,
           duc.is_value_driver
      FROM dealcloud.deal_underlying_company duc
      JOIN dealcloud.deal d ON d.id = duc.deal_id
      LEFT JOIN dealcloud.organization o_parent ON o_parent.id = d.organization_id
     WHERE duc.organization_id = ANY(org_ids)
       AND (duc.is_underlying = TRUE OR duc.is_value_driver = TRUE)
       AND d.status IN ('Portfolio Company', 'Partnership')
),
underlying_top AS (
    -- Sort dealcloud-confirmed rows ahead of llm_derived-only ones, then
    -- by date. Keeps highest-confidence positions visible first.
    SELECT * FROM underlying_all
     ORDER BY (connection_source = 'llm_derived') ASC,
              date DESC NULLS LAST, deal_id DESC
     LIMIT 5
),
-- Manager look-through: the org sits in the portfolio of a fund manager
-- that is the COUNTERPARTY of a deal we hold, per organization_portfolio
-- (investor -> portfolio edges built from that manager's OTHER deals'
-- underlying companies). Deliberately weaker than underlying_all and
-- NOT counted in in_portfolio: the source deal is often a different
-- vintage from the one we hold. Rows already in underlying_all are
-- excluded so each holding is reported once.
via_manager_all AS (
    SELECT hd.id               AS held_deal_id,
           hd.name             AS held_deal_name,
           hd.status           AS held_deal_status,
           hd.organization_id  AS manager_org_id,
           o_mgr.name          AS manager_org_name,
           op.is_value_driver,
           (SELECT jsonb_agg(jsonb_build_object(
                       'deal_id', sd.id, 'deal_name', sd.name,
                       'status', sd.status)
                     ORDER BY sd.new_deal_date DESC NULLS LAST)
              FROM dealcloud.deal sd
             WHERE sd.id = ANY(op.deal_ids)) AS source_deals
      FROM dealcloud.organization_portfolio op
      JOIN dealcloud.deal hd
           ON hd.organization_id = op.investor_org_id
          AND hd.status IN ('Portfolio Company', 'Partnership')
      LEFT JOIN dealcloud.organization o_mgr ON o_mgr.id = hd.organization_id
     WHERE op.portfolio_org_id = ANY(org_ids)
       AND NOT EXISTS (SELECT 1 FROM underlying_all u WHERE u.deal_id = hd.id)
),
via_manager_top AS (
    SELECT * FROM via_manager_all
     ORDER BY held_deal_id DESC
     LIMIT 5
),
-- Documents linked to the bundle, via organization_entity. The denorm
-- table replaces the prior alias->document_organization_alias hop.
-- DISTINCT because organization_entity is keyed on (org, type, id, alias)
-- so a doc matching multiple aliases for the same org appears N times.
target_doc_ids AS (
    SELECT DISTINCT entity_id AS doc_id
      FROM dealcloud.organization_entity
     WHERE organization_id = ANY(org_ids)
       AND entity_type = 'document'
),
-- Cap: skip docs that mention >10 distinct orgs. Those are firm-level
-- overview decks / fund summaries, not deal-specific evidence.
target_doc_ids_filtered AS (
    SELECT td.doc_id
      FROM target_doc_ids td
     WHERE (
        SELECT COUNT(DISTINCT organization_id)
          FROM dealcloud.organization_entity
         WHERE entity_type = 'document'
           AND entity_id = td.doc_id
     ) <= 10
),
dc_counterparty_orgs AS (
    SELECT DISTINCT organization_id
      FROM dealcloud.deal
     WHERE organization_id IS NOT NULL
),
hint_pairs AS (
    -- (document, co-mentioned DC counterparty) pairs, post-cap. We
    -- DISTINCT on (doc, other_org) because the same other_org may match
    -- the doc through multiple aliases.
    SELECT DISTINCT td.doc_id,
           oe_other.organization_id AS co_org_id,
           o_other.name             AS co_org_name
      FROM target_doc_ids_filtered td
      JOIN dealcloud.organization_entity oe_other
           ON oe_other.entity_type = 'document'
          AND oe_other.entity_id   = td.doc_id
      JOIN dealcloud.organization o_other
           ON o_other.id = oe_other.organization_id
     WHERE oe_other.organization_id IN (SELECT organization_id FROM dc_counterparty_orgs)
       AND NOT (oe_other.organization_id = ANY(org_ids))
),
hint_top AS (
    SELECT hp.doc_id        AS document_id,
           doc.name          AS document_name,
           hp.co_org_id      AS co_mentioned_dc_org_id,
           hp.co_org_name    AS co_mentioned_dc_org_name,
           LEFT(COALESCE(doc.summary, ''), 200) AS context_snippet
      FROM hint_pairs hp
      JOIN dealcloud.document doc ON doc.id = hp.doc_id
     ORDER BY doc.modified_at DESC NULLS LAST
     LIMIT 3
)
SELECT jsonb_build_object(
    'in_portfolio',
        ((SELECT COUNT(*) FROM counterparty_all) > 0
         OR (SELECT COUNT(*) FROM underlying_all) > 0),
    'as_counterparty', jsonb_build_object(
        'count', (SELECT COUNT(*) FROM counterparty_all),
        'deals', COALESCE(
            (SELECT jsonb_agg(to_jsonb(c)) FROM counterparty_top c),
            '[]'::jsonb)
    ),
    'as_underlying', jsonb_build_object(
        'count', (SELECT COUNT(*) FROM underlying_all),
        'deals', COALESCE(
            (SELECT jsonb_agg(to_jsonb(u)) FROM underlying_top u),
            '[]'::jsonb)
    ),
    'via_counterparty_manager_portfolio', jsonb_build_object(
        'count', (SELECT COUNT(*) FROM via_manager_all),
        'deals', COALESCE(
            (SELECT jsonb_agg(to_jsonb(v)) FROM via_manager_top v),
            '[]'::jsonb)
    ),
    'doc_only_underlying_hints', jsonb_build_object(
        'count', (SELECT COUNT(*) FROM hint_pairs),
        'samples', COALESCE(
            (SELECT jsonb_agg(to_jsonb(h)) FROM hint_top h),
            '[]'::jsonb)
    )
);
$function$
