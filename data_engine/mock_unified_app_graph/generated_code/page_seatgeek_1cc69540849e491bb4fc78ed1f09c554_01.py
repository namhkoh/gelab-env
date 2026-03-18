# page_id: page_seatgeek_1cc69540849e491bb4fc78ed1f09c554_01
# screenshot: 2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4.png
# step_index: 1/7
# task: Open SeatGeek. Search "Madison Square Garden". Select the next upcoming event. Who are the performers of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background, status bar, headers, section card backgrounds, dividers, and bottom nav bar.
# Uses provided variables: canvas (PIL Image 1440x2960 RGB) and draw (ImageDraw)

W, H = canvas.size

# Colors
status_bar_color = (245, 246, 247)        # very light gray for status area
header_bg = (255, 255, 255)               # white header (canvas already white, but explicit)
divider_gray = (230, 230, 230)            # subtle divider
hero_start = (22, 56, 123)                # deep blue (left/top of hero card)
hero_end   = (81, 120, 181)               # lighter blue (right/bottom of hero card)
hero_shadow = (220, 227, 240)             # soft shadow under hero card
card_radius = 28

# 1) Status bar area at top (~72px)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Thin separator below status bar
draw.line([(16, status_h), (W-16, status_h)], fill=divider_gray, width=1)

# 2) Header region (location & small subtitle). Keep it white but add bottom divider.
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (W, header_bottom)], fill=header_bg)
# bottom divider under header
draw.line([(16, header_bottom), (W-16, header_bottom)], fill=divider_gray, width=1)

# 3) Large hero card background (rounded rectangle with vertical gradient)
hero_x = 48
hero_y = 360
hero_w = 1344
hero_h = 840
hero_bbox = [hero_x, hero_y, hero_x + hero_w, hero_y + hero_h]
# Draw soft shadow below and to the right
shadow_offset = 10
shadow_bbox = [hero_bbox[0]+shadow_offset, hero_bbox[1]+shadow_offset,
               hero_bbox[2]+shadow_offset, hero_bbox[3]+shadow_offset]
draw.rounded_rectangle(shadow_bbox, radius=card_radius+2, fill=hero_shadow)
# Gradient fill: top->bottom blending hero_start to hero_end
top = hero_bbox[1]
bottom = hero_bbox[3]
for i in range(hero_h):
    t = i / max(hero_h - 1, 1)
    r = int(hero_start[0] * (1 - t) + hero_end[0] * t)
    g = int(hero_start[1] * (1 - t) + hero_end[1] * t)
    b = int(hero_start[2] * (1 - t) + hero_end[2] * t)
    y = top + i
    # draw a horizontal line across hero area for the gradient
    draw.line([(hero_bbox[0], y), (hero_bbox[2], y)], fill=(r, g, b))
# Draw rounded mask/clip by drawing white rounded rectangle inside then composite effect:
# To create crisp rounded corners, overpaint corners with white then draw rounded rectangle fill.
# Re-draw rounded rect border as slightly lighter for subtle edge
draw.rounded_rectangle(hero_bbox, radius=card_radius, outline=(255,255,255), width=2)

# 4) Light divider between hero and next content (above "Just for you")
just_for_you_y = 1310  # approximate vertical position for section header
divider_y = int((hero_bbox[3] + just_for_you_y) / 2)
draw.line([(16, divider_y), (W-16, divider_y)], fill=divider_gray, width=1)

# 5) "Just for you" section: subtle grouping background (keeps main background white)
# Provide a faint rounded rectangle behind the row area (so the pasted thumbnails appear on it)
jfu_group_top = 1360
jfu_group_left = 16
jfu_group_right = W - 16
jfu_group_height = 220
jfu_group_bbox = [jfu_group_left, jfu_group_top, jfu_group_right, jfu_group_top + jfu_group_height]
draw.rounded_rectangle(jfu_group_bbox, radius=18, fill=(255,255,255), outline=(245,245,245), width=1)

# 6) Separator line under the "Just for you" cards before trending header
under_jfu = jfu_group_bbox[3] + 36
draw.line([(16, under_jfu), (W-16, under_jfu)], fill=divider_gray, width=1)

# 7) Trending events header area (white, with subtle left padding)
trending_top = under_jfu + 30
trending_bottom = trending_top + 72
draw.rectangle([(0, trending_top), (W, trending_bottom)], fill=header_bg)
# Divider under trending header
draw.line([(16, trending_bottom), (W-16, trending_bottom)], fill=divider_gray, width=1)

# 8) List row separators for the trending events list
# Detected event rows occupy areas starting around y ~2190, 2430, 2670. Add separators aligned to that.
row_separators = [2188, 2428, 2668]  # approximate y positions for separators between rows
for y in row_separators:
    draw.line([(16, y), (W-16, y)], fill=divider_gray, width=1)

# 9) Left decorative dotted/badges column background (faint circular backgrounds for ranking)
# We'll draw subtle pale circles at positions corresponding to the numbered badges, but NOT draw numbers.
badge_color = (255, 239, 241)  # pale peach
badge_positions = [(64, 2140), (64, 2380), (64, 2620)]
for (cx, cy) in badge_positions:
    r = 48
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=badge_color)

# 10) Bottom navigation bar area with top divider (leave icons to be pasted)
nav_top = 2792
draw.line([(0, nav_top), (W, nav_top)], fill=divider_gray, width=1)
draw.rectangle([(0, nav_top), (W, H)], fill=(255,255,255))

# 11) Additional subtle right-edge visual hint (a vertical accent on the far right, as in screenshot)
accent_x = W - 16
draw.line([(accent_x, header_bottom + 8), (accent_x, H - 160)], fill=(248, 241, 255), width=8)

# 12) Final subtle top-left content area shadow under header (to separate header from main content)
draw.rectangle([(0, header_bottom), (W, header_bottom+6)], fill=(250,250,250))

# Done drawing structural/background elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/00_icon_Clippers.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Clippers"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/01_icon_Dodger_Stadium.png
try:
    _c1 = get_crop(1, 1309, 236)
    canvas.paste(_c1, (0, 2197), _c1)
except Exception:
    pass
layout["Dodger_Stadium"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 99, 151)
    canvas.paste(_c2, (1341, 2243), _c2)
except Exception:
    pass
layout["View_all"] = [1341, 2243, 1440, 2394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/03_icon_S262.png
try:
    _c3 = get_crop(3, 462, 519)
    canvas.paste(_c3, (48, 1431), _c3)
except Exception:
    pass
layout["S262+"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/04_icon_Angel_Stadium_of_Anaheim.png
try:
    _c4 = get_crop(4, 1309, 236)
    canvas.paste(_c4, (0, 2433), _c4)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/05_icon_Los_Angeles_CA.png
try:
    _c5 = get_crop(5, 61, 58)
    canvas.paste(_c5, (243, 5), _c5)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/06_icon_7_44_Wy.png
try:
    _c6 = get_crop(6, 54, 54)
    canvas.paste(_c6, (115, 7), _c6)
except Exception:
    pass
layout["7:44_Wy"] = [115, 7, 169, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/07_icon_7_44_Wy.png
try:
    _c7 = get_crop(7, 47, 56)
    canvas.paste(_c7, (185, 6), _c7)
except Exception:
    pass
layout["7:44_Wy"] = [185, 6, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/08_icon_888.png
try:
    _c8 = get_crop(8, 97, 63)
    canvas.paste(_c8, (1216, 1), _c8)
except Exception:
    pass
layout["888"] = [1216, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 63)
    canvas.paste(_c9, (1320, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1320, 2, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/10_icon_888.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/11_icon_Tracking.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (864, 2792), _c11)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 102, 150)
    canvas.paste(_c12, (1338, 2480), _c12)
except Exception:
    pass
layout["icon_12"] = [1338, 2480, 1440, 2630]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 58)
    canvas.paste(_c13, (314, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [314, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/14_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (288, 2792), _c14)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/15_icon_7_PM.png
try:
    _c15 = get_crop(15, 264, 183)
    canvas.paste(_c15, (1176, 2014), _c15)
except Exception:
    pass
layout["7_PM"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 45, 65)
    canvas.paste(_c16, (1155, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1155, 1, 1200, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/17_icon_S65.png
try:
    _c17 = get_crop(17, 462, 533)
    canvas.paste(_c17, (546, 1431), _c17)
except Exception:
    pass
layout["S65+"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/18_icon_Browse.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/19_icon_W_Conf_Ist_Rnd.png
try:
    _c19 = get_crop(19, 462, 533)
    canvas.paste(_c19, (546, 1431), _c19)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/20_icon_7_44_Wy.png
try:
    _c20 = get_crop(20, 93, 60)
    canvas.paste(_c20, (14, 2), _c20)
except Exception:
    pass
layout["7:44_Wy"] = [14, 2, 107, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 100, 118)
    canvas.paste(_c21, (1340, 2707), _c21)
except Exception:
    pass
layout["icon_21"] = [1340, 2707, 1440, 2825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 116, 127)
    canvas.paste(_c22, (1138, 2495), _c22)
except Exception:
    pass
layout["icon_22"] = [1138, 2495, 1254, 2622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/23_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (576, 2792), _c23)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/24_icon_Los_Angeles_CA.png
try:
    _c24 = get_crop(24, 461, 84)
    canvas.paste(_c24, (40, 122), _c24)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [40, 122, 501, 206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/25_icon_Account.png
try:
    _c25 = get_crop(25, 288, 168)
    canvas.paste(_c25, (1152, 2792), _c25)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/26_text_date.png
try:
    _c26 = get_crop(26, 114, 52)
    canvas.paste(_c26, (137, 208), _c26)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/27_text_Just_for_you.png
try:
    _c27 = get_crop(27, 309, 66)
    canvas.paste(_c27, (38, 1310), _c27)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 347, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/28_text_View_all.png
try:
    _c28 = get_crop(28, 264, 183)
    canvas.paste(_c28, (1176, 1248), _c28)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/29_text_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c29 = get_crop(29, 288, 168)
    canvas.paste(_c29, (576, 2792), _c29)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 396, 519)
    canvas.paste(_c30, (1044, 1431), _c30)
except Exception:
    pass
layout["Tracking"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (408, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_01_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-4/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (906, 1455), _c32)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
