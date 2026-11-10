"""Agent tools backed by the client-owned REST API."""

from typing import Any, Dict, List, Optional
from urllib.parse import quote

from tau2.environment.toolkit import ToolType, is_tool
from tau2.hyper.client_api import ClientAPIToolKitBase


class Tools(ClientAPIToolKitBase):
    """Customer-support operations exposed to the agent."""

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        response = self.client_api.request(method, path, body=body)
        response.raise_for_status()
        return response.body

    @is_tool(ToolType.READ)
    def search_customer(
        self,
        email: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        postal_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find a customer using an email or complete first_name, last_name, and postal_code."""
        payload: Dict[str, Any] = {}
        if email is not None:
            payload["email"] = email
        else:
            if not (first_name and last_name and postal_code):
                raise ValueError(
                    "Provide either email or first_name, last_name, and postal_code."
                )
            payload["first_name"] = first_name
            payload["last_name"] = last_name
            payload["postal_code"] = postal_code
        return self._request("POST", "/v1/customers/search", payload)

    @is_tool(ToolType.READ)
    def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve a verified customer profile."""
        identifier = quote(customer_id, safe="")
        return self._request("GET", f"/v1/customers/{identifier}")

    @is_tool(ToolType.READ)
    def list_products(self) -> Dict[str, Any]:
        """List all catalog products."""
        return self._request("GET", "/v1/catalog/products")

    @is_tool(ToolType.READ)
    def get_product(self, product_id: str) -> Dict[str, Any]:
        """Retrieve a catalog product and its purchasable item variants."""
        identifier = quote(product_id, safe="")
        return self._request("GET", f"/v1/catalog/products/{identifier}")

    @is_tool(ToolType.READ)
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Retrieve an order by its customer-facing identifier."""
        identifier = quote(order_id, safe="")
        return self._request("GET", f"/v1/orders/{identifier}")

    @is_tool(ToolType.WRITE)
    def update_default_address(
        self,
        customer_id: str,
        address_line_1: str,
        city: str,
        region: str,
        country: str,
        postal_code: str,
        address_line_2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Replace the customer's default delivery address."""
        identifier = quote(customer_id, safe="")
        return self._request(
            "PUT",
            f"/v1/customers/{identifier}/default-shipping-address",
            {
                "address_line_1": address_line_1,
                "address_line_2": address_line_2,
                "city": city,
                "region": region,
                "country": country,
                "postal_code": postal_code,
            },
        )

    @is_tool(ToolType.WRITE)
    def update_order_address(
        self,
        order_id: str,
        address_line_1: str,
        city: str,
        region: str,
        country: str,
        postal_code: str,
        address_line_2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Replace a pending order's shipping address."""
        identifier = quote(order_id, safe="")
        return self._request(
            "PUT",
            f"/v1/orders/{identifier}/shipping-address",
            {
                "address_line_1": address_line_1,
                "address_line_2": address_line_2,
                "city": city,
                "region": region,
                "country": country,
                "postal_code": postal_code,
            },
        )

    @is_tool(ToolType.WRITE)
    def update_order_payment_method(
        self,
        order_id: str,
        payment_method_id: str,
    ) -> Dict[str, Any]:
        """Replace a pending order's payment method."""
        identifier = quote(order_id, safe="")
        return self._request(
            "PUT",
            f"/v1/orders/{identifier}/payment-method",
            {"payment_method_id": payment_method_id},
        )

    @is_tool(ToolType.WRITE)
    def cancel_order(
        self,
        order_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Cancel a pending order with a valid reason ('no longer needed' or 'ordered by mistake')."""
        identifier = quote(order_id, safe="")
        return self._request(
            "POST",
            f"/v1/orders/{identifier}/cancellations",
            {"reason": reason},
        )

    @is_tool(ToolType.WRITE)
    def modify_order_items(
        self,
        order_id: str,
        replacements: List[Dict[str, str]],
        payment_method_id: str,
    ) -> Dict[str, Any]:
        """Modify items in a pending order. Each replacement is {'existing_item_id': ..., 'replacement_item_id': ...}."""
        identifier = quote(order_id, safe="")
        return self._request(
            "POST",
            f"/v1/orders/{identifier}/item-modifications",
            {"replacements": replacements, "payment_method_id": payment_method_id},
        )

    @is_tool(ToolType.WRITE)
    def create_return_request(
        self,
        order_id: str,
        item_ids: List[str],
        refund_payment_method_id: str,
    ) -> Dict[str, Any]:
        """Submit a confirmed return request for a delivered order."""
        identifier = quote(order_id, safe="")
        return self._request(
            "POST",
            f"/v1/orders/{identifier}/returns",
            {
                "item_ids": item_ids,
                "refund_payment_method_id": refund_payment_method_id,
            },
        )

    @is_tool(ToolType.WRITE)
    def create_exchange_request(
        self,
        order_id: str,
        replacements: List[Dict[str, str]],
        payment_method_id: str,
    ) -> Dict[str, Any]:
        """Submit a confirmed same-product exchange request. Each replacement is {'existing_item_id': ..., 'replacement_item_id': ...}."""
        identifier = quote(order_id, safe="")
        return self._request(
            "POST",
            f"/v1/orders/{identifier}/exchanges",
            {"replacements": replacements, "payment_method_id": payment_method_id},
        )

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> Dict[str, Any]:
        """Transfer the conversation to a human support representative."""
        conversation_id = quote(
            self.client_api.context.conversation_id,
            safe="",
        )
        return self._request(
            "POST",
            f"/v1/conversations/{conversation_id}/transfers",
            {"summary": summary},
        )
