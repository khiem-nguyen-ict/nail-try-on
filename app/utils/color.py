def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert a hex color string (e.g. '#FF0000' or 'FF0000') to an RGB tuple."""
    RED = (255, 0, 0)
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return RED
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return RED
