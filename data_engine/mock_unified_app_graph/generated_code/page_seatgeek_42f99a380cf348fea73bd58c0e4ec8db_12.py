# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_12
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15.png
# step_index: 12/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structural background and layout for the mobile UI (1440x2960)
# Variables available: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (subtle light gray)
draw.rectangle((0, 0, 1440, 2960), fill="#f2f3f4")

# Status bar area at top (~50px) - slightly darker than background
STATUS_BAR_H = 50
draw.rectangle((0, 0, 1440, STATUS_BAR_H), fill="#d0d3d6")

# Modal / main sheet background with rounded top corners (starts below status bar)
MODAL_LEFT = 20
MODAL_TOP = STATUS_BAR_H
MODAL_RIGHT = 1440 - 20
MODAL_BOTTOM = 2960 - 40
MODAL_RADIUS = 32
draw.rounded_rectangle(
    (MODAL_LEFT, MODAL_TOP, MODAL_RIGHT, MODAL_BOTTOM),
    radius=MODAL_RADIUS,
    fill="#ffffff",
)

# Subtle shadow line under modal header (to simulate elevation)
draw.line((MODAL_LEFT, MODAL_TOP + MODAL_RADIUS, MODAL_RIGHT, MODAL_TOP + MODAL_RADIUS), fill="#ececed", width=2)

# Header toolbar area inside modal (where "Filters" header sits) - implied background by spacing,
# add a faint divider below it
HEADER_TOP = MODAL_TOP + 18
HEADER_BOTTOM = HEADER_TOP + 120
draw.rectangle((MODAL_LEFT + 10, HEADER_TOP, MODAL_RIGHT - 10, HEADER_BOTTOM), fill="#ffffff")
draw.line((MODAL_LEFT + 20, HEADER_BOTTOM, MODAL_RIGHT - 20, HEADER_BOTTOM), fill="#ededee", width=1)

# Quantity section container separator lines
QUANTITY_TOP = HEADER_BOTTOM + 28
QUANTITY_BOTTOM = QUANTITY_TOP + 140
# top separator
draw.line((MODAL_LEFT + 20, QUANTITY_TOP, MODAL_RIGHT - 20, QUANTITY_TOP), fill="#f0f0f0", width=1)
# bottom separator
draw.line((MODAL_LEFT + 20, QUANTITY_BOTTOM, MODAL_RIGHT - 20, QUANTITY_BOTTOM), fill="#ededee", width=1)

# Price per ticket section area separators
PRICE_TOP = QUANTITY_BOTTOM + 24
PRICE_BOTTOM = PRICE_TOP + 640
# top separator
draw.line((MODAL_LEFT + 20, PRICE_TOP, MODAL_RIGHT - 20, PRICE_TOP), fill="#f0f0f0", width=1)
# subtle divider nearer the slider area
SLIDER_MARK_Y = PRICE_TOP + 320
draw.line((MODAL_LEFT + 30, SLIDER_MARK_Y, MODAL_RIGHT - 30, SLIDER_MARK_Y), fill="#fafafa", width=1)

# Draw a neutral slider track (without handles) to indicate range control (no icons drawn)
TRACK_LEFT = MODAL_LEFT + 120
TRACK_RIGHT = MODAL_RIGHT - 120
TRACK_Y = SLIDER_MARK_Y + 60
TRACK_HEIGHT = 6
draw.rounded_rectangle((TRACK_LEFT, TRACK_Y - TRACK_HEIGHT // 2, TRACK_RIGHT, TRACK_Y + TRACK_HEIGHT // 2), radius=3, fill="#e9e9ea")

# "Show prices with fees" row separator
FEES_ROW_Y = SLIDER_MARK_Y + 140
draw.line((MODAL_LEFT + 20, FEES_ROW_Y, MODAL_RIGHT - 20, FEES_ROW_Y), fill="#ededee", width=1)

# Options / Sort section card (separate block)
OPTIONS_TOP = FEES_ROW_Y + 60
OPTIONS_BOTTOM = OPTIONS_TOP + 260
# light divider above
draw.line((MODAL_LEFT + 20, OPTIONS_TOP, MODAL_RIGHT - 20, OPTIONS_TOP), fill="#f3f3f3", width=1)
# thin bottom separator of the options group
draw.line((MODAL_LEFT + 20, OPTIONS_BOTTOM, MODAL_RIGHT - 20, OPTIONS_BOTTOM), fill="#efefef", width=1)

# Sort by row separator inside options area
SORT_ROW_Y = OPTIONS_TOP + 80
draw.line((MODAL_LEFT + 20, SORT_ROW_Y, MODAL_RIGHT - 20, SORT_ROW_Y), fill="#fafafa", width=1)

# Large empty content area below for any additional controls (maintain clean white background)
CONTENT_TOP = OPTIONS_BOTTOM + 20
CONTENT_BOTTOM = MODAL_BOTTOM - 120
draw.rectangle((MODAL_LEFT + 10, CONTENT_TOP, MODAL_RIGHT - 10, CONTENT_BOTTOM), fill="#ffffff")

# Bottom action area (dock) with subtle top divider and shadow
DOCK_TOP = MODAL_BOTTOM - 90
DOCK_BOTTOM = MODAL_BOTTOM
draw.rectangle((0, DOCK_TOP, 1440, DOCK_BOTTOM), fill="#ffffff")
# Top divider
draw.line((20, DOCK_TOP, 1420, DOCK_TOP), fill="#e8e8e8", width=1)
# light shadow band above dock to suggest elevation
draw.rectangle((0, DOCK_TOP - 8, 1440, DOCK_TOP - 2), fill="#fafafa")

# Final very subtle vertical separators for visual grouping in the modal
sep_x1 = MODAL_LEFT + 180
sep_x2 = MODAL_RIGHT - 180
draw.line((sep_x1, HEADER_BOTTOM + 6, sep_x1, CONTENT_BOTTOM - 6), fill="#fbfbfb", width=1)
draw.line((sep_x2, HEADER_BOTTOM + 6, sep_x2, CONTENT_BOTTOM - 6), fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/00_icon_3_tickets.png
try:
    _c0 = get_crop(0, 283, 110)
    canvas.paste(_c0, (582, 512), _c0)
except Exception:
    pass
layout["3_tickets"] = [582, 512, 865, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/01_icon_View_2_listings.png
try:
    _c1 = get_crop(1, 422, 144)
    canvas.paste(_c1, (958, 2768), _c1)
except Exception:
    pass
layout["View_2_listings"] = [958, 2768, 1380, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/02_icon_3_tickets.png
try:
    _c2 = get_crop(2, 144, 110)
    canvas.paste(_c2, (893, 512), _c2)
except Exception:
    pass
layout["3_tickets"] = [893, 512, 1037, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/03_icon_Any.png
try:
    _c3 = get_crop(3, 176, 110)
    canvas.paste(_c3, (60, 512), _c3)
except Exception:
    pass
layout["Any"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/04_icon_6.png
try:
    _c4 = get_crop(4, 144, 110)
    canvas.paste(_c4, (1219, 512), _c4)
except Exception:
    pass
layout["6"] = [1219, 512, 1363, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/05_icon_5.png
try:
    _c5 = get_crop(5, 144, 110)
    canvas.paste(_c5, (1056, 512), _c5)
except Exception:
    pass
layout["5"] = [1056, 512, 1200, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/06_icon_3_tickets.png
try:
    _c6 = get_crop(6, 144, 110)
    canvas.paste(_c6, (412, 512), _c6)
except Exception:
    pass
layout["3_tickets"] = [412, 512, 556, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/07_icon_Any.png
try:
    _c7 = get_crop(7, 144, 110)
    canvas.paste(_c7, (257, 512), _c7)
except Exception:
    pass
layout["Any"] = [257, 512, 401, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/08_icon_GEEK.png
try:
    _c8 = get_crop(8, 53, 59)
    canvas.paste(_c8, (183, 2), _c8)
except Exception:
    pass
layout["GEEK"] = [183, 2, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/09_icon_GEEK.png
try:
    _c9 = get_crop(9, 57, 56)
    canvas.paste(_c9, (245, 5), _c9)
except Exception:
    pass
layout["GEEK"] = [245, 5, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 45, 63)
    canvas.paste(_c10, (1156, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1156, 2, 1201, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/11_icon_7.42_my.png
try:
    _c11 = get_crop(11, 56, 61)
    canvas.paste(_c11, (113, 2), _c11)
except Exception:
    pass
layout["7.42_my"] = [113, 2, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 59, 118)
    canvas.paste(_c12, (1381, 509), _c12)
except Exception:
    pass
layout["icon_12"] = [1381, 509, 1440, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 95, 60)
    canvas.paste(_c13, (1215, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1215, 3, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 50, 55)
    canvas.paste(_c14, (1320, 5), _c14)
except Exception:
    pass
layout["icon_14"] = [1320, 5, 1370, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 102, 103)
    canvas.paste(_c15, (557, 1347), _c15)
except Exception:
    pass
layout["icon_15"] = [557, 1347, 659, 1450]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/16_icon_clickable_10.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1251, 1500), _c16)
except Exception:
    pass
layout["clickable_10"] = [1251, 1500, 1395, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/17_icon_Price.png
try:
    _c17 = get_crop(17, 1440, 144)
    canvas.paste(_c17, (0, 1878), _c17)
except Exception:
    pass
layout["Price"] = [0, 1878, 1440, 2022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 41, 61)
    canvas.paste(_c18, (665, 1357), _c18)
except Exception:
    pass
layout["icon_18"] = [665, 1357, 706, 1418]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/19_icon_Price.png
try:
    _c19 = get_crop(19, 85, 85)
    canvas.paste(_c19, (1305, 1910), _c19)
except Exception:
    pass
layout["Price"] = [1305, 1910, 1390, 1995]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/20_icon_7.42_my.png
try:
    _c20 = get_crop(20, 122, 62)
    canvas.paste(_c20, (12, 0), _c20)
except Exception:
    pass
layout["7.42_my"] = [12, 0, 134, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/21_text_Filters.png
try:
    _c21 = get_crop(21, 1344, 156)
    canvas.paste(_c21, (48, 120), _c21)
except Exception:
    pass
layout["Filters"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/22_text_Quantity.png
try:
    _c22 = get_crop(22, 176, 110)
    canvas.paste(_c22, (60, 512), _c22)
except Exception:
    pass
layout["Quantity"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/23_text_Price_per_ticket.png
try:
    _c23 = get_crop(23, 176, 110)
    canvas.paste(_c23, (60, 512), _c23)
except Exception:
    pass
layout["Price_per_ticket"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/24_text_S414-8499.png
try:
    _c24 = get_crop(24, 1440, 139)
    canvas.paste(_c24, (0, 910), _c24)
except Exception:
    pass
layout["S414-8499"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/25_text_price_based_on_filters_is_S509.png
try:
    _c25 = get_crop(25, 1440, 139)
    canvas.paste(_c25, (0, 910), _c25)
except Exception:
    pass
layout["price_based_on_filters_is"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/26_text_Show_prices_with_fees.png
try:
    _c26 = get_crop(26, 1440, 144)
    canvas.paste(_c26, (0, 1500), _c26)
except Exception:
    pass
layout["Show_prices_with_fees"] = [0, 1500, 1440, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/27_text_Options.png
try:
    _c27 = get_crop(27, 192, 61)
    canvas.paste(_c27, (55, 1784), _c27)
except Exception:
    pass
layout["Options"] = [55, 1784, 247, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/28_text_Sort_by.png
try:
    _c28 = get_crop(28, 178, 63)
    canvas.paste(_c28, (55, 1923), _c28)
except Exception:
    pass
layout["Sort_by"] = [55, 1923, 233, 1986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_12_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-15/29_text_Clear_all.png
try:
    _c29 = get_crop(29, 193, 144)
    canvas.paste(_c29, (60, 2766), _c29)
except Exception:
    pass
layout["Clear_all"] = [60, 2766, 253, 2910]
