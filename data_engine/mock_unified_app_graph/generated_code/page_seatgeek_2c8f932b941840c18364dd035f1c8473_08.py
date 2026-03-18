# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_08
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11.png
# step_index: 8/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (PIL Image and ImageDraw)
w, h = canvas.size

# Colors
bg_color = (242, 244, 246)        # overall light bluish-gray background
status_bar_color = (228, 230, 232)  # slightly darker top status bar
header_shadow = (220, 223, 226)
header_fill = (255, 255, 255)     # white header pill
header_outline = (225, 227, 229)
map_card_bg = (239, 241, 244)     # pale card behind map
map_card_border = (217, 220, 224)
panel_bg = (255, 255, 255)        # white listings panel
thumb_bg = (233, 238, 241)        # thumbnail placeholder bg
separator = (220, 222, 224)

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (top)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)
# subtle bottom divider under status bar
draw.line([(0, status_h - 1), (w, status_h - 1)], fill=separator, width=1)

# Header / toolbar: large rounded pill container
header_margin_x = 24
header_top = 72
header_bottom = 180
header_box = [header_margin_x, header_top, w - header_margin_x, header_bottom]
# shadow behind header (slight)
shadow_offset = 6
draw.rounded_rectangle(
    [header_box[0], header_box[1] + shadow_offset, header_box[2], header_box[3] + shadow_offset],
    radius=56,
    fill=header_shadow
)
# main header pill
draw.rounded_rectangle(header_box, radius=56, fill=header_fill, outline=header_outline, width=1)

# Divider below header area
divider_y = header_bottom + 18
draw.line([(24, divider_y), (w - 24, divider_y)], fill=separator, width=1)

# Filter chips background row (just a subtle clear strip to anchor chips)
chips_top = header_bottom + 32
chips_bottom = chips_top + 96
# draw a very light rounded rect background band for chips (keeps chips readable)
draw.rounded_rectangle([24, chips_top, w - 24, chips_bottom], radius=48, fill=bg_color)

# Main seat map container card (background only)
map_margin_x = 48
map_top = chips_bottom + 20
map_bottom = 1600
map_box = [map_margin_x, map_top, w - map_margin_x, map_bottom]
# card shadow (subtle)
draw.rounded_rectangle([map_box[0], map_box[1] + 8, map_box[2], map_box[3] + 8], radius=32, fill=header_shadow)
# card background
draw.rounded_rectangle(map_box, radius=32, fill=map_card_bg, outline=map_card_border, width=2)

# Add an inner lighter inset to suggest padding for the map
inset = 18
draw.rounded_rectangle([map_box[0] + inset, map_box[1] + inset, map_box[2] - inset, map_box[3] - inset],
                       radius=20, outline=(236, 238, 240), width=1)

# Decorative large dark content area behind the stage/map (background block only)
# This is only a background area; detailed map elements will be pasted later.
stage_bg_margin = 140
draw.rounded_rectangle(
    [map_box[0] + stage_bg_margin, map_box[1] + stage_bg_margin,
     map_box[2] - stage_bg_margin, map_box[3] - stage_bg_margin],
    radius=12, fill=(245, 246, 247)
)
# subtle darker center band to indicate central area background (no labels)
center_band_h = 220
cb_top = (map_box[1] + map_box[3]) // 2 - center_band_h // 2
cb_bot = cb_top + center_band_h
draw.rectangle([map_box[0] + 40, cb_top, map_box[2] - 40, cb_bot], fill=(231, 233, 236))

# Listings panel at bottom (rounded top corners)
panel_top = 1960
panel_box = [0, panel_top, w, h]
draw.rectangle(panel_box, fill=panel_bg)
# rounded top edge (drawn by overlaying a white rounded rect for smooth corners)
draw.rounded_rectangle([16, panel_top - 12, w - 16, h], radius=28, fill=panel_bg)

# Top header bar inside listings panel (space for "96 Listings" and sort control)
list_header_h = 120
draw.line([(24, panel_top + list_header_h), (w - 24, panel_top + list_header_h)], fill=separator, width=1)

# Draw two sample listing item background cards (placeholders only — no text/icons)
item_margin_x = 24
item_w = w - item_margin_x * 2
first_item_top = panel_top + 30
item_h = 170
item_radius = 14

for i in range(2):
    top = first_item_top + i * (item_h + 26)
    bbox = [item_margin_x, top, item_margin_x + item_w, top + item_h]
    # card background
    draw.rounded_rectangle(bbox, radius=item_radius, fill=panel_bg, outline=separator, width=1)
    # left thumbnail placeholder
    thumb_margin = 20
    thumb_box = [bbox[0] + thumb_margin, bbox[1] + thumb_margin,
                 bbox[0] + thumb_margin + 260, bbox[1] + item_h - thumb_margin]
    draw.rounded_rectangle(thumb_box, radius=12, fill=thumb_bg, outline=(226, 229, 231))
    # faint inner map thumbnail indicator (just a light shape)
    inner = 18
    draw.rectangle([thumb_box[0] + inner, thumb_box[1] + inner, thumb_box[2] - inner, thumb_box[3] - inner],
                   fill=(246, 247, 249))

    # right-side text area background (keeps layout but no text)
    text_area_left = thumb_box[2] + 20
    text_area = [text_area_left, bbox[1] + 24, bbox[2] - 24, bbox[1] + item_h - 24]
    # subtle blocks representing where text will be pasted (light bars, not actual text)
    bar_h = 22
    bar_gap = 12
    # three placeholder bars stacked vertically (as background)
    draw.rectangle([text_area[0], text_area[1], text_area[2] * 0.65, text_area[1] + bar_h], fill=(250, 250, 250))
    draw.rectangle([text_area[0], text_area[1] + bar_h + bar_gap, text_area[2] * 0.5, text_area[1] + 2 * bar_h + bar_gap],
                   fill=(250, 250, 250))
    draw.rectangle([text_area[0], text_area[1] + 2 * (bar_h + bar_gap), text_area[2] * 0.4,
                    text_area[1] + 3 * bar_h + 2 * bar_gap], fill=(250, 250, 250))

    # separator under each item (except after last)
    draw.line([(item_margin_x + 16, bbox[3] + 14), (w - item_margin_x - 16, bbox[3] + 14)], fill=separator, width=1)

# Final subtle top border for the listings panel
draw.line([(0, panel_top), (w, panel_top)], fill=separator, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (542, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [542, 312, 877, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/01_icon_2_tickets.png
try:
    _c1 = get_crop(1, 266, 108)
    canvas.paste(_c1, (240, 312), _c1)
except Exception:
    pass
layout["2_tickets"] = [240, 312, 506, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/02_icon_Best_seats.png
try:
    _c2 = get_crop(2, 303, 108)
    canvas.paste(_c2, (913, 312), _c2)
except Exception:
    pass
layout["Best_seats"] = [913, 312, 1216, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/03_icon_96_Listings.png
try:
    _c3 = get_crop(3, 1440, 455)
    canvas.paste(_c3, (0, 2134), _c3)
except Exception:
    pass
layout["96_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/05_icon_8.9.png
try:
    _c5 = get_crop(5, 1440, 371)
    canvas.paste(_c5, (0, 2589), _c5)
except Exception:
    pass
layout["8.9"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/06_icon_Low_pri.png
try:
    _c6 = get_crop(6, 188, 108)
    canvas.paste(_c6, (1252, 312), _c6)
except Exception:
    pass
layout["Low_pri"] = [1252, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 64)
    canvas.paste(_c7, (1152, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1152, 1, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/08_icon_5.08_Wy.png
try:
    _c8 = get_crop(8, 67, 62)
    canvas.paste(_c8, (111, 1), _c8)
except Exception:
    pass
layout["5.08_Wy"] = [111, 1, 178, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/09_icon_6.png
try:
    _c9 = get_crop(9, 103, 63)
    canvas.paste(_c9, (1212, 1), _c9)
except Exception:
    pass
layout["6_"] = [1212, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/10_icon_5.08_Wy.png
try:
    _c10 = get_crop(10, 54, 60)
    canvas.paste(_c10, (181, 2), _c10)
except Exception:
    pass
layout["5.08_Wy"] = [181, 2, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/11_icon_Love.png
try:
    _c11 = get_crop(11, 1344, 156)
    canvas.paste(_c11, (48, 120), _c11)
except Exception:
    pass
layout["Love"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 65, 61)
    canvas.paste(_c12, (242, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [242, 2, 307, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/13_icon_6.png
try:
    _c13 = get_crop(13, 156, 156)
    canvas.paste(_c13, (1236, 120), _c13)
except Exception:
    pass
layout["6_"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 60)
    canvas.paste(_c14, (1319, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 2, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 60, 62)
    canvas.paste(_c15, (313, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 2, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 63)
    canvas.paste(_c17, (381, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [381, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/18_icon_Amazing_deal.png
try:
    _c18 = get_crop(18, 1440, 455)
    canvas.paste(_c18, (0, 2134), _c18)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/19_text_207.png
try:
    _c19 = get_crop(19, 45, 27)
    canvas.paste(_c19, (553, 721), _c19)
except Exception:
    pass
layout["207"] = [553, 721, 598, 748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/20_text_209.png
try:
    _c20 = get_crop(20, 45, 27)
    canvas.paste(_c20, (613, 712), _c20)
except Exception:
    pass
layout["209"] = [613, 712, 658, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/21_text_208.png
try:
    _c21 = get_crop(21, 48, 27)
    canvas.paste(_c21, (851, 721), _c21)
except Exception:
    pass
layout["208"] = [851, 721, 899, 748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/22_text_305.png
try:
    _c22 = get_crop(22, 48, 28)
    canvas.paste(_c22, (226, 788), _c22)
except Exception:
    pass
layout["305"] = [226, 788, 274, 816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/23_text_306.png
try:
    _c23 = get_crop(23, 46, 28)
    canvas.paste(_c23, (1186, 781), _c23)
except Exception:
    pass
layout["306"] = [1186, 781, 1232, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/24_text_205.png
try:
    _c24 = get_crop(24, 48, 29)
    canvas.paste(_c24, (351, 821), _c24)
except Exception:
    pass
layout["205"] = [351, 821, 399, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/25_text_206.png
try:
    _c25 = get_crop(25, 46, 29)
    canvas.paste(_c25, (1052, 821), _c25)
except Exception:
    pass
layout["206"] = [1052, 821, 1098, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/26_text_305.png
try:
    _c26 = get_crop(26, 46, 27)
    canvas.paste(_c26, (171, 939), _c26)
except Exception:
    pass
layout["305"] = [171, 939, 217, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/27_text_205.png
try:
    _c27 = get_crop(27, 45, 27)
    canvas.paste(_c27, (354, 932), _c27)
except Exception:
    pass
layout["205"] = [354, 932, 399, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/28_text_206.png
try:
    _c28 = get_crop(28, 46, 27)
    canvas.paste(_c28, (1052, 932), _c28)
except Exception:
    pass
layout["206"] = [1052, 932, 1098, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/29_text_306.png
try:
    _c29 = get_crop(29, 45, 27)
    canvas.paste(_c29, (1242, 932), _c29)
except Exception:
    pass
layout["306"] = [1242, 932, 1287, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/30_text_STAGE.png
try:
    _c30 = get_crop(30, 41, 16)
    canvas.paste(_c30, (706, 1203), _c30)
except Exception:
    pass
layout["STAGE"] = [706, 1203, 747, 1219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/31_text_203.png
try:
    _c31 = get_crop(31, 48, 29)
    canvas.paste(_c31, (344, 1279), _c31)
except Exception:
    pass
layout["203"] = [344, 1279, 392, 1308]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/32_text_104.png
try:
    _c32 = get_crop(32, 46, 27)
    canvas.paste(_c32, (869, 1274), _c32)
except Exception:
    pass
layout["104"] = [869, 1274, 915, 1301]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/33_text_204.png
try:
    _c33 = get_crop(33, 48, 27)
    canvas.paste(_c33, (1001, 1279), _c33)
except Exception:
    pass
layout["204"] = [1001, 1279, 1049, 1306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/34_text_303.png
try:
    _c34 = get_crop(34, 46, 27)
    canvas.paste(_c34, (180, 1364), _c34)
except Exception:
    pass
layout["303"] = [180, 1364, 226, 1391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/35_text_304.png
try:
    _c35 = get_crop(35, 46, 27)
    canvas.paste(_c35, (1232, 1371), _c35)
except Exception:
    pass
layout["304"] = [1232, 1371, 1278, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/36_text_101.png
try:
    _c36 = get_crop(36, 43, 27)
    canvas.paste(_c36, (599, 1429), _c36)
except Exception:
    pass
layout["101"] = [599, 1429, 642, 1456]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/37_text_102.png
try:
    _c37 = get_crop(37, 46, 27)
    canvas.paste(_c37, (807, 1429), _c37)
except Exception:
    pass
layout["102"] = [807, 1429, 853, 1456]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/38_text_203.png
try:
    _c38 = get_crop(38, 45, 30)
    canvas.paste(_c38, (347, 1558), _c38)
except Exception:
    pass
layout["203"] = [347, 1558, 392, 1588]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/39_text_204.png
try:
    _c39 = get_crop(39, 46, 30)
    canvas.paste(_c39, (1059, 1558), _c39)
except Exception:
    pass
layout["204"] = [1059, 1558, 1105, 1588]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/40_text_200.png
try:
    _c40 = get_crop(40, 48, 30)
    canvas.paste(_c40, (640, 1572), _c40)
except Exception:
    pass
layout["200"] = [640, 1572, 688, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/41_text_96_Listings.png
try:
    _c41 = get_crop(41, 305, 80)
    canvas.paste(_c41, (54, 2025), _c41)
except Exception:
    pass
layout["96_Listings"] = [54, 2025, 359, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/42_text_S151each.png
try:
    _c42 = get_crop(42, 1440, 371)
    canvas.paste(_c42, (0, 2589), _c42)
except Exception:
    pass
layout["S151each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/43_text_Price_includes_fees.png
try:
    _c43 = get_crop(43, 1440, 371)
    canvas.paste(_c43, (0, 2589), _c43)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/44_text_8.9.png
try:
    _c44 = get_crop(44, 50, 29)
    canvas.paste(_c44, (502, 2812), _c44)
except Exception:
    pass
layout["8.9"] = [502, 2812, 552, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/45_text_Amazing_deal.png
try:
    _c45 = get_crop(45, 1440, 371)
    canvas.paste(_c45, (0, 2589), _c45)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/46_text_2-6_tickets.png
try:
    _c46 = get_crop(46, 1440, 371)
    canvas.paste(_c46, (0, 2589), _c46)
except Exception:
    pass
layout["2-6_tickets"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_08_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-11/47_clickable_Back.png
try:
    _c47 = get_crop(47, 156, 156)
    canvas.paste(_c47, (48, 120), _c47)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
