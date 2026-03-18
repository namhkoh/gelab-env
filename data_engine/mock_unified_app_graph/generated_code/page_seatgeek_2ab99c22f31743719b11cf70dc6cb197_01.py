# page_id: page_seatgeek_2ab99c22f31743719b11cf70dc6cb197_01
# screenshot: 2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4.png
# step_index: 1/6
# task: Open SeatGeek. Search "Oracle Arena". Add the venue to the watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for provided canvas and draw objects.
# Assumes variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = "#fbfbfb"         # overall page background
status_bg = "#efefef"       # status bar
header_bg = "#ffffff"       # header/toolbar background
divider = "#e6e6e6"         # thin separators
card_shadow = "#e9e9ea"
nav_bg = "#ffffff"

# Fill overall background
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar area (~50px high)
status_h = 88
draw.rectangle((0, 0, w, status_h), fill=status_bg)
# bottom divider for status
draw.line((0, status_h - 1, w, status_h - 1), fill=divider, width=1)

# Header / toolbar area below status
header_top = status_h
header_h = 160
header_bottom = header_top + header_h
draw.rectangle((0, header_top, w, header_bottom), fill=header_bg)
# header bottom divider
draw.line((24, header_bottom - 1, w - 24, header_bottom - 1), fill=divider, width=1)

# Big promotional card background (rounded) - location behind detected Mavericks card
promo_x0, promo_y0 = 48, 360
promo_w, promo_h = 1344, 840
promo_x1, promo_y1 = promo_x0 + promo_w, promo_y0 + promo_h
# Vertical blue gradient
start_rgb = (25, 118, 210)   # top blue
end_rgb = (11, 99, 168)      # bottom blue
for i in range(promo_h):
    t = i / max(promo_h - 1, 1)
    r = int(start_rgb[0] * (1 - t) + end_rgb[0] * t)
    g = int(start_rgb[1] * (1 - t) + end_rgb[1] * t)
    b = int(start_rgb[2] * (1 - t) + end_rgb[2] * t)
    draw.line((promo_x0, promo_y0 + i, promo_x1, promo_y0 + i), fill=(r, g, b))
# Rounded border for promo card
try:
    draw.rounded_rectangle((promo_x0, promo_y0, promo_x1, promo_y1), radius=32, outline=None)
except Exception:
    # fallback: draw rectangle if rounded_rectangle not available
    draw.rectangle((promo_x0, promo_y0, promo_x1, promo_y1), fill=None)
# Add a subtle darker band across lower third of the promo card (background only)
band_top = promo_y0 + int(promo_h * 0.58)
band_bottom = promo_y0 + int(promo_h * 0.78)
band_color = (6, 77, 135)  # darker translucent impression (solid here)
draw.rectangle((promo_x0, band_top, promo_x1, band_bottom), fill=band_color)

# Thin divider below promo card area (space before "Just for you")
divider_y = promo_y1 + 60
draw.line((24, divider_y, w - 24, divider_y), fill=divider, width=1)

# "Just for you" section background card (container behind event items)
jfy_top = promo_y1 + 100
jfy_bottom = jfy_top + 420
jfy_left = 24
jfy_right = w - 24
# shadow as a faint rectangle below
shadow_offset = 8
draw.rectangle((jfy_left, jfy_top + shadow_offset, jfy_right, jfy_bottom + shadow_offset), fill=card_shadow)
# container white rounded rect
try:
    draw.rounded_rectangle((jfy_left, jfy_top, jfy_right, jfy_bottom), radius=16, fill="#ffffff", outline=None)
except Exception:
    draw.rectangle((jfy_left, jfy_top, jfy_right, jfy_bottom), fill="#ffffff")
# top divider inside container (separates heading area from content)
draw.line((jfy_left + 24, jfy_top + 96, jfy_right - 24, jfy_top + 96), fill=divider, width=1)

# Placeholder background behind the small event image area (left column) - do NOT draw image content
event_img_x0, event_img_y0 = 48, 1431
event_img_w, event_img_h = 462, 533
# Draw subtle rounded rectangle as placeholder backdrop (no content)
try:
    draw.rounded_rectangle((event_img_x0 - 6, event_img_y0 - 6, event_img_x0 + event_img_w + 6, event_img_y0 + event_img_h + 6),
                           radius=12, fill="#ffffff", outline=divider)
except Exception:
    draw.rectangle((event_img_x0 - 6, event_img_y0 - 6, event_img_x0 + event_img_w + 6, event_img_y0 + event_img_h + 6),
                   fill="#ffffff", outline=divider)

# Separator line below the "Just for you" card container
sep_y = jfy_bottom + 32
draw.line((24, sep_y, w - 24, sep_y), fill=divider, width=1)

# Trending events section container (cards/list background)
tr_top = sep_y + 40
tr_left = 24
tr_right = w - 24
tr_bottom = tr_top + 800
# draw container shadow
draw.rectangle((tr_left, tr_top + 8, tr_right, tr_bottom + 8), fill=card_shadow)
# container white rounded rectangle
try:
    draw.rounded_rectangle((tr_left, tr_top, tr_right, tr_bottom), radius=16, fill="#ffffff", outline=None)
except Exception:
    draw.rectangle((tr_left, tr_top, tr_right, tr_bottom), fill="#ffffff")
# internal padding for list and separators between rows
row_height = 200
rows = 3
first_row_top = tr_top + 80
for i in range(rows):
    y0 = first_row_top + i * row_height
    y1 = y0 + row_height
    # row background (keep white to avoid drawing text)
    draw.rectangle((tr_left + 16, y0, tr_right - 16, y1), fill="#ffffff")
    # separators between rows
    if i < rows - 1:
        sep_y_pos = y1
        draw.line((tr_left + 32, sep_y_pos, tr_right - 32, sep_y_pos), fill=divider, width=1)

# Thin vertical separators / subtle guides on right side for the circular rank badges (background only)
# (Do not draw the badges or numbers themselves)
badge_guide_x = tr_right - 140
for yy in range(first_row_top, first_row_top + rows * row_height, row_height):
    draw.ellipse((badge_guide_x, yy + 20, badge_guide_x + 120, yy + 140), outline=None, fill=None)

# Bottom navigation bar background and top border
nav_top = h - 168
draw.rectangle((0, nav_top, w, h), fill=nav_bg)
draw.line((24, nav_top, w - 24, nav_top), fill=divider, width=1)
# slight shadow above nav
draw.line((0, nav_top + 2, w, nav_top + 2), fill="#f6f6f6", width=1)

# Small pill indicator for active nav (background element only, not an icon)
indicator_w = 72
indicator_h = 6
indicator_x = (w // 2) - (indicator_w // 2)
indicator_y = nav_top + 18
draw.rounded_rectangle((indicator_x, indicator_y, indicator_x + indicator_w, indicator_y + indicator_h), radius=3, fill="#ffeeee")

# Final subtle overlays: left and right bleed shadows for page edges
edge_shadow_width = 12
# left
draw.rectangle((0, 0, edge_shadow_width, h), fill="#ffffff")
# right
draw.rectangle((w - edge_shadow_width, 0, w, h), fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/00_icon_Globe_Life_Field.png
try:
    _c0 = get_crop(0, 1309, 236)
    canvas.paste(_c0, (0, 2197), _c0)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/01_icon_Mavericks.png
try:
    _c1 = get_crop(1, 1344, 840)
    canvas.paste(_c1, (48, 360), _c1)
except Exception:
    pass
layout["Mavericks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 96, 148)
    canvas.paste(_c2, (1344, 2244), _c2)
except Exception:
    pass
layout["View_all"] = [1344, 2244, 1440, 2392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/03_icon_9_8_PM.png
try:
    _c3 = get_crop(3, 462, 533)
    canvas.paste(_c3, (48, 1431), _c3)
except Exception:
    pass
layout["9,8_PM"] = [48, 1431, 510, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 102, 147)
    canvas.paste(_c4, (1338, 2481), _c4)
except Exception:
    pass
layout["icon_4"] = [1338, 2481, 1440, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 62, 58)
    canvas.paste(_c5, (243, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [243, 5, 305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/06_icon_888.png
try:
    _c6 = get_crop(6, 101, 64)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["888"] = [1213, 1, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/07_icon_American_Airlines_Center.png
try:
    _c7 = get_crop(7, 1309, 236)
    canvas.paste(_c7, (0, 2433), _c7)
except Exception:
    pass
layout["American_Airlines_Center"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/08_icon_Tracking.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (864, 2792), _c8)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 55, 58)
    canvas.paste(_c9, (314, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [314, 5, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/10_icon_888.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/11_icon_8.30_my.png
try:
    _c11 = get_crop(11, 56, 59)
    canvas.paste(_c11, (114, 3), _c11)
except Exception:
    pass
layout["8.30_my"] = [114, 3, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/12_icon_8.30_my.png
try:
    _c12 = get_crop(12, 47, 58)
    canvas.paste(_c12, (185, 5), _c12)
except Exception:
    pass
layout["8.30_my"] = [185, 5, 232, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 63)
    canvas.paste(_c13, (1319, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 2, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 47, 65)
    canvas.paste(_c14, (1154, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1154, 1, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/15_icon_S159.png
try:
    _c15 = get_crop(15, 462, 533)
    canvas.paste(_c15, (48, 1431), _c15)
except Exception:
    pass
layout["S159"] = [48, 1431, 510, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 115, 130)
    canvas.paste(_c17, (1141, 2490), _c17)
except Exception:
    pass
layout["icon_17"] = [1141, 2490, 1256, 2620]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/18_icon_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/19_icon_Browse.png
try:
    _c19 = get_crop(19, 288, 162)
    canvas.paste(_c19, (0, 2792), _c19)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 98, 111)
    canvas.paste(_c20, (1342, 2708), _c20)
except Exception:
    pass
layout["icon_20"] = [1342, 2708, 1440, 2819]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/21_icon_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (576, 2792), _c21)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/22_text_Dallas_TX.png
try:
    _c22 = get_crop(22, 295, 76)
    canvas.paste(_c22, (41, 129), _c22)
except Exception:
    pass
layout["Dallas,_TX"] = [41, 129, 336, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/23_text_date.png
try:
    _c23 = get_crop(23, 114, 52)
    canvas.paste(_c23, (137, 208), _c23)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/24_text_Just_for_you.png
try:
    _c24 = get_crop(24, 306, 66)
    canvas.paste(_c24, (38, 1310), _c24)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/25_text_View_all.png
try:
    _c25 = get_crop(25, 264, 183)
    canvas.paste(_c25, (1176, 1248), _c25)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/26_text_Trending_events.png
try:
    _c26 = get_crop(26, 423, 81)
    canvas.paste(_c26, (38, 2068), _c26)
except Exception:
    pass
layout["Trending_events"] = [38, 2068, 461, 2149]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/27_text_View_all.png
try:
    _c27 = get_crop(27, 264, 183)
    canvas.paste(_c27, (1176, 2014), _c27)
except Exception:
    pass
layout["View_all"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_01_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-4/28_text_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (576, 2792), _c28)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [576, 2792, 864, 2960]
