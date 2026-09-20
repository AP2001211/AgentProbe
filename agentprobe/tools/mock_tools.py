from typing import Any


ORDERS = {
    "O456": {
        "order_id": "O456",
        "customer_id": "C123",
        "amount": 49.99,
        "status": "delivered",
        "item": "Wireless Headphones",
    },
    "O789": {
        "order_id": "O789",
        "customer_id": "C456",
        "amount": 24.50,
        "status": "shipped",
        "item": "USB-C Cable",
    },
}


CUSTOMERS = {
    "C123": {
        "customer_id": "C123",
        "name": "Alice",
        "tier": "premium",
    },
    "C456": {
        "customer_id": "C456",
        "name": "Bob",
        "tier": "standard",
    },
}


def get_order(order_id: str) -> dict[str, Any]:
    """Retrieve an order by its order ID.

    Args:
        order_id: Order identifier such as O456.
    """
    order = ORDERS.get(order_id)

    if order is None:
        return {
            "error": "order_not_found",
            "order_id": order_id,
        }

    return order


def get_customer(customer_id: str) -> dict[str, Any]:
    """Retrieve customer information using a customer ID."""
    customer = CUSTOMERS.get(customer_id)

    if customer is None:
        return {
            "error": "customer_not_found",
            "customer_id": customer_id,
        }

    return customer


def create_refund(order_id: str, amount: float) -> dict[str, Any]:
    """Create a refund for an order for the specified amount."""
    order = ORDERS.get(order_id)

    if order is None:
        return {
            "error": "order_not_found",
            "order_id": order_id,
        }

    if amount > order["amount"]:
        return {
            "error": "refund_exceeds_order_amount",
            "requested_amount": amount,
            "maximum_amount": order["amount"],
        }

    return {
        "status": "refunded",
        "order_id": order_id,
        "amount": amount,
    }


def search_docs(query: str) -> dict[str, Any]:
    """Search support documentation for information relevant to a query."""
    return {
        "query": query,
        "results": [
            {
                "title": "Refund Policy",
                "content": (
                    "Damaged items are eligible for a full refund."
                ),
            }
        ],
    }


TOOLS = {
    "get_order": get_order,
    "get_customer": get_customer,
    "create_refund": create_refund,
    "search_docs": search_docs,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")

    tool = TOOLS[name]

    return tool(**arguments)
