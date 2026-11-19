"""Re-export MapRenderer from map_renderer for backward compatibility."""

from app.mapping.map_renderer import (
    MapRenderer,
    RISK_COLORS_HEX,
    RISK_BG_COLORS,
)

__all__ = ["MapRenderer", "RISK_COLORS_HEX", "RISK_BG_COLORS"]
