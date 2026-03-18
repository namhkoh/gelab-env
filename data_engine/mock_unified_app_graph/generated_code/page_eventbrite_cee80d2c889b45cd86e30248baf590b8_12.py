# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_12
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14.png
# step_index: 12/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas (1440x2960) and draw are provided.
# Draw status bar background
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(245, 245, 245))

# Draw a subtle bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(230, 230, 230), width=2)

# Draw large header/banner background (yellow gradient block)
banner_top = status_h
banner_bottom = 520
# simple vertical gradient
for i in range(banner_top, banner_bottom):
    t = (i - banner_top) / max(1, banner_bottom - banner_top)
    # from bright yellow to warm yellow-orange
    r = int(255 * (1 - 0.0) + 0 * 0)
    g = int(205 * (1 - t) + 160 * t)
    b = int(0 * (1 - t) + 0 * t)
    draw.line([(0, i), (1440, i)], fill=(r, g, b))

# Add subtle vignette darker strips on left/right edges (to mimic blurred crop behind icons)
edge_width = 120
grad_steps = edge_width
for x in range(edge_width):
    alpha = int(25 * (1 - x / grad_steps))
    # left
    draw.line([(x, banner_top), (x, banner_bottom)], fill=(0 + alpha, 0 + alpha, 0 + alpha))
    # right
    xr = 1440 - 1 - x
    draw.line([(xr, banner_top), (xr, banner_bottom)], fill=(0 + alpha, 0 + alpha, 0 + alpha))

# Add a subtle bottom shadow under banner (to separate from content)
shadow_top = banner_bottom
shadow_height = 12
for i in range(shadow_height):
    alpha = int(60 * (1 - i / shadow_height))
    y = shadow_top + i
    draw.line([(48, y), (1440 - 48, y)], fill=(220 - alpha // 2, 220 - alpha // 2, 220 - alpha // 2))

# Content area background stays white; draw main organizer card background (rounded)
card_x0 = 48
card_x1 = 1392
card_y0 = 920
card_y1 = 1104
card_radius = 28
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)], radius=card_radius, fill=(246, 246, 249))

# Inside the organizer card add a subtle inner highlight to mimic slight inset
inner_pad = 10
draw.rounded_rectangle([(card_x0 + inner_pad, card_y0 + inner_pad), (card_x1 - inner_pad, card_y1 - inner_pad)],
                       radius=card_radius - 6, outline=(235, 234, 238), width=1)

# Section separator line below organizer card
sep_y = card_y1 + 80
draw.line([(48, sep_y), (1440 - 48, sep_y)], fill=(235, 234, 238), width=2)

# "About this event" section background area (subtle very light gray band)
about_top = sep_y + 40
about_bottom = about_top + 340
draw.rectangle([(48, about_top), (1392, about_bottom)], fill=(255, 255, 255))
# subtle top divider for the about section
draw.line([(48, about_top), (1392, about_top)], fill=(245, 245, 246), width=1)

# Another thin divider below the about text (separator)
about_sep = about_bottom + 28
draw.line([(48, about_sep), (1392, about_sep)], fill=(236, 235, 238), width=2)

# Location header area - leave white, but draw a faint top padding box to anchor content
loc_top = about_sep + 40
loc_bottom = loc_top + 420
# faint background band for location area
draw.rectangle([(0, loc_top), (1440, loc_top + 14)], fill=(255, 255, 255))
# subtle divider above the location details
draw.line([(48, loc_top), (1392, loc_top)], fill=(245, 244, 247), width=1)

# Sticky footer background (do not draw the button itself; only the bar behind it)
footer_h = 200
footer_top = 2960 - footer_h
# Slightly off-white panel with top border
draw.rectangle([(0, footer_top), (1440, 2960)], fill=(250, 249, 249))
draw.line([(0, footer_top), (1440, footer_top)], fill=(230, 229, 231), width=2)
# Add subtle inner shadow at top of footer
for i in range(6):
    y = footer_top + i
    shade = 240 - i * 3
    draw.line([(0, y), (1440, y)], fill=(shade, shade, shade))

# Small price pill area background (left side) - very faint
price_box_w = 300
price_box_h = 110
price_box_x = 48
price_box_y = footer_top + (footer_h - price_box_h) // 2
draw.rounded_rectangle([(price_box_x, price_box_y), (price_box_x + price_box_w, price_box_y + price_box_h)],
                       radius=18, fill=(255, 255, 255), outline=(235, 234, 238), width=1)

# Add a subtle drop shadow under the footer to visually separate from content (bottom edge)
for i in range(12):
    alpha = int(24 * (1 - i / 12))
    y = 2960 - 1 - i
    draw.line([(0, y), (1440, y)], fill=(230 - alpha, 230 - alpha, 230 - alpha))

# Additional subtle separators between content blocks
div_positions = [card_y1 + 220, about_bottom + 200, loc_top + 220]
for y in div_positions:
    draw.line([(48, y), (1392, y)], fill=(245, 244, 246), width=1)

# End of background/structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/02_icon_Food_Drink.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2205), _c2)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/03_icon_6_30PM-9_30PA.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["6:30PM-9:30PA"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 67)
    canvas.paste(_c5, (1155, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [1155, 2, 1203, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/06_icon_9.45.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["9.45"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 63)
    canvas.paste(_c7, (1327, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1327, 3, 1372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/08_icon_9.45.png
try:
    _c8 = get_crop(8, 51, 60)
    canvas.paste(_c8, (184, 3), _c8)
except Exception:
    pass
layout["9.45"] = [184, 3, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/09_icon_6.30_PM.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1116, 108), _c9)
except Exception:
    pass
layout["6.30_PM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 57)
    canvas.paste(_c10, (315, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [315, 6, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 55, 60)
    canvas.paste(_c11, (247, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [247, 4, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 100, 64)
    canvas.paste(_c12, (1214, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1214, 2, 1314, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/13_icon_Show_map.png
try:
    _c13 = get_crop(13, 226, 144)
    canvas.paste(_c13, (1166, 2423), _c13)
except Exception:
    pass
layout["Show_map"] = [1166, 2423, 1392, 2567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/14_icon_Join_us_for_this_charitable_event_with_f.png
try:
    _c14 = get_crop(14, 234, 144)
    canvas.paste(_c14, (48, 2205), _c14)
except Exception:
    pass
layout["Join_us_for_this_charitab"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/15_text_9.45.png
try:
    _c15 = get_crop(15, 94, 43)
    canvas.paste(_c15, (20, 15), _c15)
except Exception:
    pass
layout["9.45"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/16_text_The_Stone_House.png
try:
    _c16 = get_crop(16, 364, 144)
    canvas.paste(_c16, (144, 1028), _c16)
except Exception:
    pass
layout["The_Stone_House"] = [144, 1028, 508, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/17_text_87_Followers.png
try:
    _c17 = get_crop(17, 364, 144)
    canvas.paste(_c17, (144, 1028), _c17)
except Exception:
    pass
layout["87_Followers"] = [144, 1028, 508, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/18_text_The_Stone_House_at_Clove_Lakes.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1295), _c18)
except Exception:
    pass
layout["The_Stone_House_at_Clove_"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/19_text_3_hrs.png
try:
    _c19 = get_crop(19, 114, 50)
    canvas.paste(_c19, (139, 1452), _c19)
except Exception:
    pass
layout["3_hrs"] = [139, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/20_text_Refund_policy.png
try:
    _c20 = get_crop(20, 299, 63)
    canvas.paste(_c20, (138, 1558), _c20)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1295), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/22_text_Location.png
try:
    _c22 = get_crop(22, 241, 56)
    canvas.paste(_c22, (42, 2470), _c22)
except Exception:
    pass
layout["Location"] = [42, 2470, 283, 2526]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/23_text_The_Stone_House_at_Clove_Lakes.png
try:
    _c23 = get_crop(23, 234, 144)
    canvas.paste(_c23, (48, 2205), _c23)
except Exception:
    pass
layout["The_Stone_House_at_Clove_"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/24_text_The_Stone_House_at_Clove_Lakes_1150_Clov.png
try:
    _c24 = get_crop(24, 570, 144)
    canvas.paste(_c24, (822, 2768), _c24)
except Exception:
    pass
layout["The_Stone_House_at_Clove_"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_12_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-14/25_text_69.png
try:
    _c25 = get_crop(25, 101, 61)
    canvas.paste(_c25, (89, 2811), _c25)
except Exception:
    pass
layout["$69"] = [89, 2811, 190, 2872]
