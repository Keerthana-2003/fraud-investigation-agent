# Card Testing (FIN-POL-021)

Card testing is when fraudsters validate stolen card numbers with very small charges.

**Rule CT-1:** 10 or more card transactions under $5.00 within 60 minutes, across 3 or more distinct
merchants, indicates card testing. Disposition: **escalate**. Recommended actions: block and reissue
the card, decline pending authorizations, notify the customer.

Repeated small charges at a **single** transit, parking, or vending merchant are normal tap-to-pay
behavior and should be **closed**.
