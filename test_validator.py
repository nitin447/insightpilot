from tools.warehouse import connect
from tools.validator import validate_sql

con = connect()

cases = [
    ("SELECT SUM(price) FROM order_items", True),
    ("SELECT revenue FROM orders", False),                 # invented column
    ("SELECT * FROM ordrs", False),                        # typo'd table
    ("SELECT SUM(i.price) FROM orders o JOIN order_items i "
     "ON i.order_id = o.order_id", True),
    ("DROP TABLE orders", False),                          # blocked
    ("SELECT customer_naem FROM customers", False),        # typo'd column
]

for sql, expect in cases:
    ok, err = validate_sql(sql, con)
    flag = "OK " if ok == expect else "!! "
    print(f"{flag}{'valid' if ok else 'invalid'}: {sql[:55]}")
    if err:
        print(f"      {err[:150]}")
con.close()