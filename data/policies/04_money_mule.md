# Money Mule Activity (FIN-POL-030)

A money mule account receives funds from multiple unrelated senders and quickly moves them onward.

**Rule MM-1:** The disposition is **file_sar** when both hold:

1. The account shares a device or IP address with 2 or more other customer accounts (use the
   network check), and
2. 80% or more of funds received in the prior 24 hours were sent out within 24 hours.

Pass-through alone (for example salary received and rent paid to a long-standing payee) with no
shared devices is normal and should be **closed**.
