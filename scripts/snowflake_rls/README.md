# Snowflake row access policies (optional)

MedAssist.AI enforces **application-level** access with JWT roles and `org_id` / `created_by_user_id` on `CLINICAL.PATIENT_ENCOUNTER`.

For deployments that require **database-enforced** isolation, add Snowflake **row access policies** on:

- `CLINICAL.PATIENT_ENCOUNTER`
- `CLINICAL.ENCOUNTER_AUDIT` (join to encounter ownership)

Typical pattern:

1. Store `ORG_ID` and `USER_ID` in a session variable set by a secure service account after validating the JWT (not shown here—requires Snowflake external OAuth or a trusted middle tier).
2. Create policies with `CASE` expressions comparing `ORG_ID` column to `CURRENT_SESSION()` context.

Because this project uses **password-based JWT** outside Snowflake, RLS in Snowflake is optional and must be aligned with your identity architecture. Coordinate with your DBA before enabling policies in shared warehouses.
