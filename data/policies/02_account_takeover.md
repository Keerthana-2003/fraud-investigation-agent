# Account Takeover (FIN-POL-014)

Indicators, evaluated within 24 hours before the alerted transaction:

1. Login from a device first seen less than 24 hours earlier.
2. A password reset or MFA change on the account.
3. A payee added less than 24 hours before a transfer to that payee.

**Rule ATO-1:** If indicator 3 is present together with indicator 1 or 2, and the transfer exceeds
$2,000, the disposition is **escalate**. Recommended actions: freeze outbound transfers, revoke
sessions on the new device, contact the customer through a verified channel.

A new device alone, with no credential change and no new payee, is not sufficient.
