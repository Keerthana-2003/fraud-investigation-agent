# Using the Transaction Risk Model (FIN-POL-060)

The transaction risk score (0-1) is decision support, not a decision. Scores above 0.8 are high
risk, 0.5-0.8 medium, below 0.5 low. Always confirm the drivers (top contributing features) against
the underlying records.

A high score alone is not sufficient for escalation. If no fraud rule (ATO-1, CT-1, MM-1, ST-1) is
met and the records match a documented legitimate pattern (FIN-POL-012, 017, 021, 030, 041), close
the alert and note the score. If no rule is met and no legitimate explanation is found, escalate for
manual review.
