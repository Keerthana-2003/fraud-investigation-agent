# Alert Triage Overview (FIN-POL-001)

Every alert must end with exactly one disposition:

- **close**: activity is explained by legitimate behavior. Document the explanation.
- **escalate**: likely fraud against the customer (account takeover, card compromise). Send to the
  Fraud Operations queue for customer contact and account controls within 4 hours.
- **file_sar**: suspicious activity by or through the customer (money laundering, structuring,
  mule activity). Route to the BSA/AML team to prepare a Suspicious Activity Report.

Investigators must cite the evidence (transactions, logins, device records, model scores, policy
sections) that supports each statement in the case file. Statements without evidence are not allowed.
