# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_11
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14.png
# step_index: 11/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant off-white)
draw.rectangle((0, 0, 1440, 2960), fill="#f6f6f6")

# Status bar area (top light grey strip)
draw.rectangle((0, 0, 1440, 60), fill="#e9eaeb")

# Modal sheet with rounded top corners (white)
sheet_margin = 24
sheet_top = 40
draw.rounded_rectangle(
    (sheet_margin, sheet_top, 1440 - sheet_margin, 2960),
    radius=32,
    fill="#ffffff",
    outline=None,
)

# Top center handle (drag indicator)
handle_w = 160
handle_h = 10
handle_x0 = (1440 - handle_w) // 2
handle_y0 = sheet_top + 14
draw.rounded_rectangle(
    (handle_x0, handle_y0, handle_x0 + handle_w, handle_y0 + handle_h),
    radius=6,
    fill="#e6e6e6",
)

# Subtle header divider line under the title area
header_divider_y = 220
draw.line((sheet_margin + 8, header_divider_y, 1440 - sheet_margin - 8, header_divider_y), fill="#efefef", width=1)

# Section separators (thin dividers between major groups)
separators = [520, 920, 1280, 1720, 2640]
for y in separators:
    draw.line((sheet_margin + 8, y, 1440 - sheet_margin - 8, y), fill="#f0f0f0", width=1)

# Subtle card outlines for main sections (Quantity row, Price card, Options card)
# Quantity section card (behind the row of circular quantity tokens)
qty_card_top = 420
qty_card_bottom = 640
draw.rounded_rectangle(
    (sheet_margin + 20, qty_card_top, 1440 - sheet_margin - 20, qty_card_bottom),
    radius=14,
    fill="#ffffff",
    outline="#f3f3f3",
    width=1
)

# Price per ticket card (contains range, slider area)
price_card_top = 700
price_card_bottom = 1240
draw.rounded_rectangle(
    (sheet_margin + 20, price_card_top, 1440 - sheet_margin - 20, price_card_bottom),
    radius=14,
    fill="#ffffff",
    outline="#f3f3f3",
    width=1
)

# Toggle/Options area card
options_card_top = 1440
options_card_bottom = 1840
draw.rounded_rectangle(
    (sheet_margin + 20, options_card_top, 1440 - sheet_margin - 20, options_card_bottom),
    radius=14,
    fill="#ffffff",
    outline="#f3f3f3",
    width=1
)

# Bottom sticky area background (subtle elevation strip behind bottom controls)
bottom_strip_top = 2680
draw.rectangle((0, bottom_strip_top, 1440, 2960), fill="#ffffff")
# soft top border to separate from content
draw.line((sheet_margin + 8, bottom_strip_top, 1440 - sheet_margin - 8, bottom_strip_top), fill="#ececec", width=1)

# Very subtle vignette / shadow at the very bottom to suggest elevation
shadow_top = 2840
for i, alpha in enumerate([10, 20, 35, 55, 80], start=0):
    y = shadow_top + i * 8
    shade = 240 - i * 6
    shade_hex = "#{:02x}{:02x}{:02x}".format(max(0, shade), max(0, shade), max(0, shade))
    draw.rectangle((0, y, 1440, y + 8), fill=shade_hex)

# Light vertical divider on the right edge of main content for subtle framing
draw.line((1440 - sheet_margin - 2, sheet_top + 8, 1440 - sheet_margin - 2, bottom_strip_top - 8), fill="#f6f6f6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/00_icon_3_tickets.png
try:
    _c0 = get_crop(0, 283, 110)
    canvas.paste(_c0, (582, 512), _c0)
except Exception:
    pass
layout["3_tickets"] = [582, 512, 865, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/01_icon_listings.png
try:
    _c1 = get_crop(1, 422, 144)
    canvas.paste(_c1, (958, 2768), _c1)
except Exception:
    pass
layout["listings"] = [958, 2768, 1380, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/02_icon_Any.png
try:
    _c2 = get_crop(2, 176, 110)
    canvas.paste(_c2, (60, 512), _c2)
except Exception:
    pass
layout["Any"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/03_icon_3_tickets.png
try:
    _c3 = get_crop(3, 144, 110)
    canvas.paste(_c3, (893, 512), _c3)
except Exception:
    pass
layout["3_tickets"] = [893, 512, 1037, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/04_icon_6.png
try:
    _c4 = get_crop(4, 144, 110)
    canvas.paste(_c4, (1219, 512), _c4)
except Exception:
    pass
layout["6"] = [1219, 512, 1363, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/05_icon_5.png
try:
    _c5 = get_crop(5, 144, 110)
    canvas.paste(_c5, (1056, 512), _c5)
except Exception:
    pass
layout["5"] = [1056, 512, 1200, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/06_icon_3_tickets.png
try:
    _c6 = get_crop(6, 144, 110)
    canvas.paste(_c6, (412, 512), _c6)
except Exception:
    pass
layout["3_tickets"] = [412, 512, 556, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/07_icon_Any.png
try:
    _c7 = get_crop(7, 144, 110)
    canvas.paste(_c7, (257, 512), _c7)
except Exception:
    pass
layout["Any"] = [257, 512, 401, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 101, 102)
    canvas.paste(_c8, (1277, 1346), _c8)
except Exception:
    pass
layout["icon_8"] = [1277, 1346, 1378, 1448]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/09_icon_GEEK.png
try:
    _c9 = get_crop(9, 53, 59)
    canvas.paste(_c9, (183, 2), _c9)
except Exception:
    pass
layout["GEEK"] = [183, 2, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/10_icon_GEEK.png
try:
    _c10 = get_crop(10, 56, 56)
    canvas.paste(_c10, (246, 5), _c10)
except Exception:
    pass
layout["GEEK"] = [246, 5, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 45, 62)
    canvas.paste(_c11, (1155, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1155, 3, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/12_icon_7.42_my.png
try:
    _c12 = get_crop(12, 56, 60)
    canvas.paste(_c12, (113, 2), _c12)
except Exception:
    pass
layout["7.42_my"] = [113, 2, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 58, 118)
    canvas.paste(_c13, (1382, 509), _c13)
except Exception:
    pass
layout["icon_13"] = [1382, 509, 1440, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 95, 59)
    canvas.paste(_c14, (1215, 4), _c14)
except Exception:
    pass
layout["icon_14"] = [1215, 4, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 53)
    canvas.paste(_c15, (1321, 6), _c15)
except Exception:
    pass
layout["icon_15"] = [1321, 6, 1370, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/16_icon_Price.png
try:
    _c16 = get_crop(16, 1440, 144)
    canvas.paste(_c16, (0, 1878), _c16)
except Exception:
    pass
layout["Price"] = [0, 1878, 1440, 2022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/17_icon_clickable_10.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1251, 1500), _c17)
except Exception:
    pass
layout["clickable_10"] = [1251, 1500, 1395, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/18_icon_7.42_my.png
try:
    _c18 = get_crop(18, 123, 62)
    canvas.paste(_c18, (13, 0), _c18)
except Exception:
    pass
layout["7.42_my"] = [13, 0, 136, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/19_icon_Price.png
try:
    _c19 = get_crop(19, 97, 112)
    canvas.paste(_c19, (1298, 1898), _c19)
except Exception:
    pass
layout["Price"] = [1298, 1898, 1395, 2010]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/20_text_Filters.png
try:
    _c20 = get_crop(20, 1344, 156)
    canvas.paste(_c20, (48, 120), _c20)
except Exception:
    pass
layout["Filters"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/21_text_Quantity.png
try:
    _c21 = get_crop(21, 176, 110)
    canvas.paste(_c21, (60, 512), _c21)
except Exception:
    pass
layout["Quantity"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/22_text_Price_per_ticket.png
try:
    _c22 = get_crop(22, 176, 110)
    canvas.paste(_c22, (60, 512), _c22)
except Exception:
    pass
layout["Price_per_ticket"] = [60, 512, 236, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/23_text_S414-8654.png
try:
    _c23 = get_crop(23, 1440, 139)
    canvas.paste(_c23, (0, 910), _c23)
except Exception:
    pass
layout["S414-8654"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/24_text_price_based_on_filters_is_S509.png
try:
    _c24 = get_crop(24, 1440, 139)
    canvas.paste(_c24, (0, 910), _c24)
except Exception:
    pass
layout["price_based_on_filters_is"] = [0, 910, 1440, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/25_text_Show_prices_with_fees.png
try:
    _c25 = get_crop(25, 1440, 144)
    canvas.paste(_c25, (0, 1500), _c25)
except Exception:
    pass
layout["Show_prices_with_fees"] = [0, 1500, 1440, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/26_text_Options.png
try:
    _c26 = get_crop(26, 192, 61)
    canvas.paste(_c26, (55, 1784), _c26)
except Exception:
    pass
layout["Options"] = [55, 1784, 247, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/27_text_Sort_by.png
try:
    _c27 = get_crop(27, 178, 63)
    canvas.paste(_c27, (55, 1923), _c27)
except Exception:
    pass
layout["Sort_by"] = [55, 1923, 233, 1986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_11_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-14/28_text_Clear_all.png
try:
    _c28 = get_crop(28, 193, 144)
    canvas.paste(_c28, (60, 2766), _c28)
except Exception:
    pass
layout["Clear_all"] = [60, 2766, 253, 2910]
