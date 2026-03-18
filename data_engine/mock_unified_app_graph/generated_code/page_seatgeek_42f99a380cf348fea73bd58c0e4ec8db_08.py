# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_08
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11.png
# step_index: 8/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background base
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~80px) - light gray to match screenshot's subtle status background
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill=(240, 241, 243))

# Modal / sheet background (rounded) that contains the UI content
sheet_top = status_h
sheet_bottom = 2700
sheet_radius = 36
draw.rounded_rectangle([(0, sheet_top), (1440, sheet_bottom)], radius=sheet_radius, fill=(255, 255, 255), outline=None)

# Subtle top shadow line for the sheet (gives separation from the status bar)
draw.line([(0, sheet_top), (1440, sheet_top)], fill=(230, 230, 230), width=1)

# Header divider under the "Filters" header area
header_div_y = sheet_top + 180  # around where the header divider sits in screenshot
draw.line([(24, header_div_y), (1440-24, header_div_y)], fill=(235, 235, 235), width=1)

# Divider under the quantity controls (separates Quantity from Price section)
quantity_div_y = 720
draw.line([(24, quantity_div_y), (1440-24, quantity_div_y)], fill=(235, 235, 235), width=1)

# Price section area - draw a soft colored dome behind the price slider (decorative background)
# Dome as an ellipse to emulate the red/orange distribution shape (keeps clear of exact handles/icons)
dome_bbox = (100, 980, 1340, 1220)
draw.ellipse(dome_bbox, fill=(255, 108, 92))  # coral/red-orange fill

# Slider track (thin dark line) across the dome (do NOT draw slider handles)
slider_y = 1120
draw.line([(80, slider_y), (1360, slider_y)], fill=(30, 30, 30), width=8)

# Divider under the price / slider area (above the "Show prices with fees" toggle region)
price_div_y = 1500
draw.line([(24, price_div_y), (1440-24, price_div_y)], fill=(235, 235, 235), width=1)

# Divider above the "Options" area to separate sections
options_div_y = 1740
draw.line([(24, options_div_y), (1440-24, options_div_y)], fill=(235, 235, 235), width=1)

# Large subtle divider near the bottom to separate content from bottom controls
bottom_div_y = 2720
draw.line([(0, bottom_div_y), (1440, bottom_div_y)], fill=(240, 240, 240), width=1)

# Bottom area subtle fade / background (simulate elevation where CTA sits)
draw.rectangle([(0, bottom_div_y), (1440, 2960)], fill=(250, 250, 250))

# Small inner card backgrounds for the Options row and Sort row (rounded rectangles, no text/icons)
opt_card_margin_x = 24
opt_card_radius = 12
# Options card area (behind "Sort by" region)
draw.rounded_rectangle([(opt_card_margin_x, options_div_y + 24), (1440 - opt_card_margin_x, options_div_y + 160)],
                       radius=opt_card_radius, fill=(255, 255, 255), outline=(245, 245, 245))

# Light separators inside content area to guide layout (thin subtle lines)
for y in (sheet_top + 340, sheet_top + 520, sheet_top + 920, sheet_top + 1320):
    draw.line([(24, y), (1440-24, y)], fill=(246, 246, 246), width=1)

# Decorative subtle vignette along top of sheet (thin darker top band)
draw.rectangle([(0, sheet_top), (1440, sheet_top + 6)], fill=(240, 240, 240))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/00_icon_3_tickets.png
try:
    _c0 = get_crop(0, 283, 110)
    canvas.paste(_c0, (582, 512), _c0)
except Exception:
    pass
layout["3_tickets"] = [582, 512, 865, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/01_icon_View_36_listings.png
try:
    _c1 = get_crop(1, 456, 144)
    canvas.paste(_c1, (924, 2768), _c1)
except Exception:
    pass
layout["View_36_listings"] = [924, 2768, 1380, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/02_icon_3_tickets.png
try:
    _c2 = get_crop(2, 144, 110)
    canvas.paste(_c2, (893, 512), _c2)
except Exception:
    pass
layout["3_tickets"] = [893, 512, 1037, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/03_icon_Any.png
try:
    _c3 = get_crop(3, 176, 110)
    canvas.paste(_c3, (60, 512), _c3)
except Exception:
    pass
layout["Any"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/04_icon_6.png
try:
    _c4 = get_crop(4, 144, 110)
    canvas.paste(_c4, (1219, 512), _c4)
except Exception:
    pass
layout["6"] = [1219, 512, 1363, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/05_icon_5.png
try:
    _c5 = get_crop(5, 144, 110)
    canvas.paste(_c5, (1056, 512), _c5)
except Exception:
    pass
layout["5"] = [1056, 512, 1200, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/06_icon_3_tickets.png
try:
    _c6 = get_crop(6, 144, 110)
    canvas.paste(_c6, (412, 512), _c6)
except Exception:
    pass
layout["3_tickets"] = [412, 512, 556, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/07_icon_Any.png
try:
    _c7 = get_crop(7, 144, 110)
    canvas.paste(_c7, (257, 512), _c7)
except Exception:
    pass
layout["Any"] = [257, 512, 401, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/08_icon_GEEK.png
try:
    _c8 = get_crop(8, 57, 56)
    canvas.paste(_c8, (245, 5), _c8)
except Exception:
    pass
layout["GEEK"] = [245, 5, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/09_icon_GEEK.png
try:
    _c9 = get_crop(9, 53, 59)
    canvas.paste(_c9, (183, 2), _c9)
except Exception:
    pass
layout["GEEK"] = [183, 2, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/10_icon_7.41_my.png
try:
    _c10 = get_crop(10, 58, 61)
    canvas.paste(_c10, (112, 1), _c10)
except Exception:
    pass
layout["7.41_my"] = [112, 1, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 45, 62)
    canvas.paste(_c11, (1155, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1155, 3, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 101, 102)
    canvas.paste(_c12, (1277, 1346), _c12)
except Exception:
    pass
layout["icon_12"] = [1277, 1346, 1378, 1448]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 58, 118)
    canvas.paste(_c13, (1382, 509), _c13)
except Exception:
    pass
layout["icon_13"] = [1382, 509, 1440, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 97, 59)
    canvas.paste(_c14, (1215, 4), _c14)
except Exception:
    pass
layout["icon_14"] = [1215, 4, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 50, 54)
    canvas.paste(_c15, (1320, 5), _c15)
except Exception:
    pass
layout["icon_15"] = [1320, 5, 1370, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 102, 102)
    canvas.paste(_c16, (56, 1346), _c16)
except Exception:
    pass
layout["icon_16"] = [56, 1346, 158, 1448]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/17_icon_Price.png
try:
    _c17 = get_crop(17, 1440, 144)
    canvas.paste(_c17, (0, 1878), _c17)
except Exception:
    pass
layout["Price"] = [0, 1878, 1440, 2022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/18_icon_clickable_10.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1251, 1500), _c18)
except Exception:
    pass
layout["clickable_10"] = [1251, 1500, 1395, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/19_text_Filters.png
try:
    _c19 = get_crop(19, 1344, 156)
    canvas.paste(_c19, (48, 120), _c19)
except Exception:
    pass
layout["Filters"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/20_text_Quantity.png
try:
    _c20 = get_crop(20, 176, 110)
    canvas.paste(_c20, (60, 512), _c20)
except Exception:
    pass
layout["Quantity"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/21_text_Price_per_ticket.png
try:
    _c21 = get_crop(21, 176, 110)
    canvas.paste(_c21, (60, 512), _c21)
except Exception:
    pass
layout["Price_per_ticket"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/22_text_S252-S654.png
try:
    _c22 = get_crop(22, 1440, 139)
    canvas.paste(_c22, (0, 910), _c22)
except Exception:
    pass
layout["S252-S654"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/23_text_price_based_on_filters_is_S423.png
try:
    _c23 = get_crop(23, 1440, 139)
    canvas.paste(_c23, (0, 910), _c23)
except Exception:
    pass
layout["price_based_on_filters_is"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/24_text_Show_prices_with_fees.png
try:
    _c24 = get_crop(24, 1440, 144)
    canvas.paste(_c24, (0, 1500), _c24)
except Exception:
    pass
layout["Show_prices_with_fees"] = [0, 1500, 1440, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/25_text_Options.png
try:
    _c25 = get_crop(25, 192, 61)
    canvas.paste(_c25, (55, 1784), _c25)
except Exception:
    pass
layout["Options"] = [55, 1784, 247, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/26_text_Sort_by.png
try:
    _c26 = get_crop(26, 178, 63)
    canvas.paste(_c26, (55, 1923), _c26)
except Exception:
    pass
layout["Sort_by"] = [55, 1923, 233, 1986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_08_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-11/27_text_Clear_all.png
try:
    _c27 = get_crop(27, 193, 144)
    canvas.paste(_c27, (60, 2766), _c27)
except Exception:
    pass
layout["Clear_all"] = [60, 2766, 253, 2910]
