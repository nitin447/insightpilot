from tools.warehouse import run_sql, preview

print("--- quarterly revenue, South electronics, freight ratio ---")
print(preview(run_sql("""
SELECT DATE_TRUNC('quarter', o.order_purchase_ts) AS quarter,
       ROUND(SUM(i.price)/1e6, 2) AS total_rev_M,
       ROUND(SUM(CASE WHEN c.customer_region='South'
                       AND p.product_category='electronics'
                      THEN i.price END)/1e6, 2) AS south_elec_M,
       ROUND(SUM(i.freight_value)/SUM(i.price), 4) AS freight_ratio
FROM orders o
JOIN order_items i ON i.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = i.product_id
WHERE o.order_status <> 'canceled'
GROUP BY 1 ORDER BY 1
""")))

print("\n--- late deliveries vs review scores ---")
print(preview(run_sql("""
SELECT DATE_TRUNC('quarter', o.order_purchase_ts) AS quarter,
       ROUND(AVG(CASE WHEN o.order_delivered_ts > o.order_estimated_delivery_ts
                      THEN 1.0 ELSE 0 END), 3) AS late_rate,
       ROUND(AVG(r.review_score), 2) AS avg_review
FROM orders o
LEFT JOIN reviews r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY 1 ORDER BY 1
""")))