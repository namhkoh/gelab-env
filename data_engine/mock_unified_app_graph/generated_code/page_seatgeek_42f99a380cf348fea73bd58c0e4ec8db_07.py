# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_07
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10.png
# step_index: 7/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: 1440x2960 RGB Image, draw: ImageDraw)
# Fonts provided: font_sm, font_md, font_lg, font_xl

# Colors
bg_outer = (247, 247, 247)      # overall app background
status_bg = (238, 238, 238)     # status bar background
modal_bg = (255, 255, 255)      # main sheet background
divider = (230, 230, 230)       # separators
muted = (245, 245, 245)         # subtle panels
accent_pale = (255, 240, 238)   # pale accent behind graph
shadow_line = (220, 220, 220)

w, h = canvas.size

# Fill outer background
draw.rectangle([(0, 0), (w, h)], fill=bg_outer)

# Status bar (top area)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=status_bg)

# Rounded sheet/modal that holds the filters content
sheet_margin = 24
sheet_top = 64
sheet_bottom = h - 140
sheet_radius = 36
draw.rounded_rectangle(
    [(sheet_margin, sheet_top), (w - sheet_margin, sheet_bottom)],
    radius=sheet_radius,
    fill=modal_bg
)

# Soft shadow/divider under header area
header_div_y = sheet_top + 84
draw.line([(sheet_margin + 12, header_div_y), (w - sheet_margin - 12, header_div_y)], fill=divider, width=1)

# Horizontal separators between major sections
separators = [
    sheet_top + 260,   # below quantity chips
    sheet_top + 560,   # below price per ticket area / slider
    sheet_top + 980,   # above options area
    sheet_top + 1240   # mid content separator
]
for y in separators:
    draw.line([(sheet_margin + 12, y), (w - sheet_margin - 12, y)], fill=divider, width=1)

# Pale accent panel behind the price-distribution graph area (do not draw graph itself)
graph_top = sheet_top + 340
graph_bottom = graph_top + 320
graph_left = sheet_margin + 40
graph_right = w - sheet_margin - 40
draw.rounded_rectangle(
    [(graph_left, graph_top), (graph_right, graph_bottom)],
    radius=24,
    fill=accent_pale
)
# add a thin divider line representing the slider track baseline (background only)
track_y = graph_bottom - 48
draw.line([(graph_left + 28, track_y), (graph_right - 28, track_y)], fill=shadow_line, width=6)

# Subtle circular background "handles" positions (background only, actual handles will be pasted)
# Draw faint ring outlines (only background ring shapes, small)
handle_r = 40
left_handle_center = (graph_left + 28, track_y)
right_handle_center = (graph_right - 28, track_y)
draw.ellipse([
    (left_handle_center[0] - handle_r, left_handle_center[1] - handle_r),
    (left_handle_center[0] + handle_r, left_handle_center[1] + handle_r)],
    fill=muted, outline=divider)
draw.ellipse([
    (right_handle_center[0] - handle_r, right_handle_center[1] - handle_r),
    (right_handle_center[0] + handle_r, right_handle_center[1] + handle_r)],
    fill=muted, outline=divider)

# Options / list item background panels
option_panel_top = sheet_top + 920
option_panel_height = 160
draw.rectangle([(sheet_margin + 12, option_panel_top), (w - sheet_margin - 12, option_panel_top + option_panel_height)], fill=modal_bg)
draw.line([(sheet_margin + 12, option_panel_top), (w - sheet_margin - 12, option_panel_top)], fill=divider, width=1)

# Additional large white content area below options (keeps the page airy)
content_area_top = option_panel_top + option_panel_height + 24
draw.rectangle([(sheet_margin + 6, content_area_top), (w - sheet_margin - 6, sheet_bottom - 60)], fill=modal_bg)

# Bottom sticky area background (behind "Clear all" and "View ... listings" controls)
bottom_bar_top = h - 196
draw.rectangle([(0, bottom_bar_top), (w, h)], fill=modal_bg)
# top border for bottom bar
draw.line([(12, bottom_bar_top), (w - 12, bottom_bar_top)], fill=divider, width=1)

# Slight vignette/shadow at the top edge of the bottom bar to separate it
draw.line([(12, bottom_bar_top + 2), (w - 12, bottom_bar_top + 2)], fill=(245,245,245), width=4)

# Final subtle edges/shadows on sheet sides to give depth
edge_shadow_width = 12
# left shadow
draw.rectangle([(sheet_margin - edge_shadow_width, sheet_top + 6), (sheet_margin, sheet_bottom - 6)], fill=(250,250,250))
# right shadow
draw.rectangle([(w - sheet_margin, sheet_top + 6), (w - sheet_margin + edge_shadow_width, sheet_bottom - 6)], fill=(250,250,250))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/00_icon_View_106_listings.png
try:
    _c0 = get_crop(0, 477, 144)
    canvas.paste(_c0, (903, 2768), _c0)
except Exception:
    pass
layout["View_106_listings"] = [903, 2768, 1380, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/01_icon_Any.png
try:
    _c1 = get_crop(1, 176, 110)
    canvas.paste(_c1, (60, 512), _c1)
except Exception:
    pass
layout["Any"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/02_icon_5.png
try:
    _c2 = get_crop(2, 144, 110)
    canvas.paste(_c2, (899, 512), _c2)
except Exception:
    pass
layout["5"] = [899, 512, 1043, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/03_icon_6.png
try:
    _c3 = get_crop(3, 144, 110)
    canvas.paste(_c3, (1062, 512), _c3)
except Exception:
    pass
layout["6"] = [1062, 512, 1206, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/04_icon_4.png
try:
    _c4 = get_crop(4, 144, 110)
    canvas.paste(_c4, (736, 512), _c4)
except Exception:
    pass
layout["4"] = [736, 512, 880, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/05_icon_7.png
try:
    _c5 = get_crop(5, 144, 110)
    canvas.paste(_c5, (1223, 512), _c5)
except Exception:
    pass
layout["7"] = [1223, 512, 1367, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/06_icon_3.png
try:
    _c6 = get_crop(6, 144, 110)
    canvas.paste(_c6, (573, 512), _c6)
except Exception:
    pass
layout["3"] = [573, 512, 717, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/07_icon_Any.png
try:
    _c7 = get_crop(7, 144, 110)
    canvas.paste(_c7, (412, 512), _c7)
except Exception:
    pass
layout["Any"] = [412, 512, 556, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/08_icon_Any.png
try:
    _c8 = get_crop(8, 144, 110)
    canvas.paste(_c8, (257, 512), _c8)
except Exception:
    pass
layout["Any"] = [257, 512, 401, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/09_icon_GEEK.png
try:
    _c9 = get_crop(9, 58, 56)
    canvas.paste(_c9, (245, 5), _c9)
except Exception:
    pass
layout["GEEK"] = [245, 5, 303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/10_icon_GEEK.png
try:
    _c10 = get_crop(10, 53, 59)
    canvas.paste(_c10, (183, 2), _c10)
except Exception:
    pass
layout["GEEK"] = [183, 2, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/11_icon_7.41_my.png
try:
    _c11 = get_crop(11, 59, 61)
    canvas.paste(_c11, (111, 1), _c11)
except Exception:
    pass
layout["7.41_my"] = [111, 1, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 63)
    canvas.paste(_c12, (1155, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1155, 3, 1200, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 101, 102)
    canvas.paste(_c13, (1277, 1346), _c13)
except Exception:
    pass
layout["icon_13"] = [1277, 1346, 1378, 1448]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 97, 60)
    canvas.paste(_c14, (1215, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1215, 3, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 55, 119)
    canvas.paste(_c15, (1385, 509), _c15)
except Exception:
    pass
layout["icon_15"] = [1385, 509, 1440, 628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 54)
    canvas.paste(_c16, (1320, 5), _c16)
except Exception:
    pass
layout["icon_16"] = [1320, 5, 1370, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/17_icon_Price.png
try:
    _c17 = get_crop(17, 1440, 144)
    canvas.paste(_c17, (0, 1878), _c17)
except Exception:
    pass
layout["Price"] = [0, 1878, 1440, 2022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/18_icon_clickable_11.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1251, 1500), _c18)
except Exception:
    pass
layout["clickable_11"] = [1251, 1500, 1395, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/19_icon_Show_prices_with_fees.png
try:
    _c19 = get_crop(19, 1440, 144)
    canvas.paste(_c19, (0, 1500), _c19)
except Exception:
    pass
layout["Show_prices_with_fees"] = [0, 1500, 1440, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/20_icon_Filters.png
try:
    _c20 = get_crop(20, 1344, 156)
    canvas.paste(_c20, (48, 120), _c20)
except Exception:
    pass
layout["Filters"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/21_text_Quantity.png
try:
    _c21 = get_crop(21, 176, 110)
    canvas.paste(_c21, (60, 512), _c21)
except Exception:
    pass
layout["Quantity"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/22_text_Price_per_ticket.png
try:
    _c22 = get_crop(22, 176, 110)
    canvas.paste(_c22, (60, 512), _c22)
except Exception:
    pass
layout["Price_per_ticket"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/23_text_S124-81_163.png
try:
    _c23 = get_crop(23, 1440, 139)
    canvas.paste(_c23, (0, 910), _c23)
except Exception:
    pass
layout["S124-81,163"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/24_text_price_based_on_filters_is_S410.png
try:
    _c24 = get_crop(24, 1440, 139)
    canvas.paste(_c24, (0, 910), _c24)
except Exception:
    pass
layout["price_based_on_filters_is"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/25_text_Show_prices_with_fees.png
try:
    _c25 = get_crop(25, 1440, 144)
    canvas.paste(_c25, (0, 1500), _c25)
except Exception:
    pass
layout["Show_prices_with_fees"] = [0, 1500, 1440, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/26_text_Options.png
try:
    _c26 = get_crop(26, 192, 61)
    canvas.paste(_c26, (55, 1784), _c26)
except Exception:
    pass
layout["Options"] = [55, 1784, 247, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/27_text_Sort_by.png
try:
    _c27 = get_crop(27, 178, 63)
    canvas.paste(_c27, (55, 1923), _c27)
except Exception:
    pass
layout["Sort_by"] = [55, 1923, 233, 1986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_07_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-10/28_text_Clear_all.png
try:
    _c28 = get_crop(28, 193, 144)
    canvas.paste(_c28, (60, 2766), _c28)
except Exception:
    pass
layout["Clear_all"] = [60, 2766, 253, 2910]
