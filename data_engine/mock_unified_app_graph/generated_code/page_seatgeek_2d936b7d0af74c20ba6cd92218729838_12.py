# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_12
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15.png
# step_index: 12/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas provided)
draw.rectangle([(0, 0), canvas.size], fill=(255, 255, 255))

# Top hero/background band (stadium photo area)
hero_top = 0
hero_bottom = 430
hero_color = (11, 34, 64)  # deep blue/navy to evoke the arena image background
draw.rectangle([(0, hero_top), (1440, hero_bottom)], fill=hero_color)

# Subtle vertical gradient in the hero area to mimic photo darkness at top
# (painted as a few horizontal strips)
for i, alpha in enumerate([0.06, 0.10, 0.14, 0.18, 0.22, 0.28]):
    y0 = hero_top + int(i * (hero_bottom - hero_top) / 6)
    y1 = hero_top + int((i + 1) * (hero_bottom - hero_top) / 6)
    # slightly lighter overlay stripes
    overlay_color = (
        int(hero_color[0] + (255 - hero_color[0]) * alpha),
        int(hero_color[1] + (255 - hero_color[1]) * alpha),
        int(hero_color[2] + (255 - hero_color[2]) * alpha),
    )
    draw.rectangle([(0, y0), (1440, y1)], fill=overlay_color)

# Status bar area (top strip for time/signal icons) - dark overlay
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill=(6, 18, 28))

# Thin divider line at bottom of hero image to separate image and content
divider_y = hero_bottom
draw.line([(24, divider_y), (1440 - 24, divider_y)], fill=(230, 230, 230), width=2)

# Subtle drop shadow under the hero band
shadow_y0 = divider_y
shadow_y1 = divider_y + 12
for i in range(6):
    alpha = int(12 - i * 2)
    if alpha <= 0:
        continue
    y = shadow_y0 + i * 2
    draw.line([(24, y), (1440 - 24, y)], fill=(220, 220, 220), width=1)

# Main content background area (keeps white but add a very soft gray full-width band
# behind where the empty-state card sits so content pasted on top reads well)
card_margin_x = 48
card_top = 980
card_bottom = 1880
card_radius = 28
card_fill = (250, 250, 250)  # subtle off-white card background
card_outline = (240, 240, 240)

# Rounded "card" area for the empty state / upcoming games section
draw.rounded_rectangle(
    [(card_margin_x, card_top), (1440 - card_margin_x, card_bottom)],
    radius=card_radius,
    fill=card_fill,
    outline=card_outline,
    width=1
)

# Add a subtle inner highlight band near the top edge of the card to suggest elevation
highlight_top = card_top + 8
draw.rounded_rectangle(
    [(card_margin_x + 8, highlight_top), (1440 - card_margin_x - 8, highlight_top + 48)],
    radius=16,
    fill=(255, 255, 255),
    outline=None
)

# Section separator lines for content areas below the card
sep_x1 = 24
sep_x2 = 1440 - 24
for y in (card_top - 80, card_bottom + 36):
    draw.line([(sep_x1, y), (sep_x2, y)], fill=(245, 245, 245), width=1)

# Small faint center guide line (very subtle) to help placement of pasted elements
# (low-contrast so it doesn't duplicate any UI elements)
center_x = canvas.size[0] // 2
draw.line([(center_x, card_top - 120), (center_x, card_bottom + 120)], fill=(250, 250, 250), width=1)

# Bottom area remains simple white; add a faint footer divider near the bottom
footer_y = canvas.size[1] - 120
draw.line([(24, footer_y), (1440 - 24, footer_y)], fill=(245, 245, 245), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/00_icon_Share_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1260, 84), _c0)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/01_icon_Track_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1104, 84), _c1)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/02_icon_6.54.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 84), _c2)
except Exception:
    pass
layout["6.54"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/03_icon_6.54.png
try:
    _c3 = get_crop(3, 56, 62)
    canvas.paste(_c3, (117, 5), _c3)
except Exception:
    pass
layout["6.54"] = [117, 5, 173, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/04_icon_6.54.png
try:
    _c4 = get_crop(4, 59, 61)
    canvas.paste(_c4, (183, 4), _c4)
except Exception:
    pass
layout["6.54"] = [183, 4, 242, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/05_icon_6.54.png
try:
    _c5 = get_crop(5, 95, 67)
    canvas.paste(_c5, (13, 3), _c5)
except Exception:
    pass
layout["6.54"] = [13, 3, 108, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 59, 64)
    canvas.paste(_c6, (244, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [244, 4, 303, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 54, 67)
    canvas.paste(_c7, (1151, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1151, 1, 1205, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 60)
    canvas.paste(_c8, (312, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [312, 5, 365, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 109, 63)
    canvas.paste(_c9, (1204, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1204, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/10_icon_Golden_State_Warriors.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1104, 84), _c10)
except Exception:
    pass
layout["Golden_State_Warriors"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 66, 62)
    canvas.paste(_c11, (1304, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1304, 2, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/12_text_No_upcoming_Games.png
try:
    _c12 = get_crop(12, 548, 75)
    canvas.paste(_c12, (447, 1515), _c12)
except Exception:
    pass
layout["No_upcoming_Games"] = [447, 1515, 995, 1590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/13_text_Track_Golden_State_Warriors_for_event_up.png
try:
    _c13 = get_crop(13, 1060, 77)
    canvas.paste(_c13, (192, 1606), _c13)
except Exception:
    pass
layout["Track_Golden_State_Warrio"] = [192, 1606, 1252, 1683]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_12_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-15/14_text_1Zt.png
try:
    _c14 = get_crop(14, 154, 142)
    canvas.paste(_c14, (646, 1289), _c14)
except Exception:
    pass
layout["1Zt"] = [646, 1289, 800, 1431]
