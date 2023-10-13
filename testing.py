parking_tickets = []

try:
    k = max(parking_tickets)
except ValueError:
    k = -1
most_tickets = k if 0 <= k else 0

print(most_tickets)
