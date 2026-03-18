# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_01
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4.png
# step_index: 1/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structure for the provided canvas
# Assumes: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), font_sm/font_md/font_lg/font_xl available

w, h = canvas.size

# Colors
bg = (249, 250, 250)            # very light off-white page background
status_bg = (241, 242, 243)     # status bar background
divider = (228, 230, 232)       # subtle dividers
card_bg = (255, 255, 255)       # white cards / sections
muted = (245, 246, 246)         # slightly different panel
nav_top_div = (220, 222, 223)   # nav top divider
accent_soft = (255, 242, 238)   # slight warm accent for list badges background (very faint)

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bg)
# subtle bottom divider under status bar
draw.line([(0, status_h - 1), (w, status_h - 1)], fill=divider, width=1)

# Header / toolbar area (below status bar)
header_top = status_h
header_h = 248  # header area ends before the big banner starts (~360)
draw.rectangle([(0, header_top), (w, header_top + header_h)], fill=card_bg)
# Header bottom divider
header_bottom_y = header_top + header_h
draw.line([(24, header_bottom_y), (w - 24, header_bottom_y)], fill=divider, width=1)

# Space reserved for the large promotional/banner card (DO NOT draw its content)
# We draw a faint rounded outline to indicate the banner container boundary only,
# using a very subtle stroke so we don't duplicate the banner's graphics.
banner_x0 = 48
banner_y0 = 360
banner_w = 1344
banner_h = 840
banner_box = (banner_x0 - 4, banner_y0 - 4, banner_x0 + banner_w + 4, banner_y0 + banner_h + 4)
draw.rounded_rectangle(banner_box, radius=24, outline=(240,240,240), width=1)

# Divider below the large banner (separates banner from next section)
banner_bottom_y = banner_y0 + banner_h
draw.line([(24, banner_bottom_y + 20), (w - 24, banner_bottom_y + 20)], fill=divider, width=1)

# "Just for you" / small-cards area
# Reserve a subtle container background behind the carousel of small cards (but do not draw card content)
just_for_you_top = 1310
just_for_you_height = 640  # covers the small-cards row down to approx 1950
jfu_box = (18, just_for_you_top - 18, w - 18, just_for_you_top + just_for_you_height)
# Slightly different fill to separate from page
draw.rectangle(jfu_box, fill=muted)
# thin top separator for the section
draw.line([(24, just_for_you_top - 18), (w - 24, just_for_you_top - 18)], fill=divider, width=1)
# thin bottom separator for the section (above trending)
draw.line([(24, just_for_you_top + just_for_you_height - 4), (w - 24, just_for_you_top + just_for_you_height - 4)], fill=divider, width=1)

# Trending events section header area
trending_top = just_for_you_top + just_for_you_height + 24  # approx 1980+
draw.rectangle([(0, trending_top - 24), (w, trending_top + 80)], fill=card_bg)
# header underline
draw.line([(24, trending_top + 80), (w - 24, trending_top + 80)], fill=divider, width=1)

# Trending list container (rounded white card)
list_left = 18
list_right = w - 18
list_top = trending_top + 96
list_bottom = 2700  # bottom of trending list area
list_rect = (list_left, list_top, list_right, list_bottom)
draw.rounded_rectangle(list_rect, radius=12, fill=card_bg, outline=(240,240,240), width=1)

# Draw separators between trending list items (positions approximated to match crops)
# Use the detected y positions for the items as visual separators:
sep_y1 = 2183  # after first item
sep_y2 = 2419  # after second item
sep_y3 = 2655  # after third item
for y in (sep_y1, sep_y2, sep_y3):
    # Draw separator only across the content area inset
    draw.line([(list_left + 24, y), (list_right - 24, y)], fill=divider, width=1)

# Soft circular placeholder backgrounds on left side of each list row (do not place icons/text)
# These are faint pastel circles to hint numeric badges/backgrounds.
badge_r = 44
badge_centers = [(list_left + 56, 2080), (list_left + 56, 2316), (list_left + 56, 2552)]
for cx, cy in badge_centers:
    draw.ellipse([(cx - badge_r, cy - badge_r), (cx + badge_r, cy + badge_r)], fill=accent_soft)

# Right-side small floating indicators (do not draw icons) - faint circular placeholders
right_badge_centers = [(list_right - 56, 2080), (list_right - 56, 2316), (list_right - 56, 2552)]
for cx, cy in right_badge_centers:
    draw.ellipse([(cx - badge_r, cy - badge_r), (cx + badge_r, cy + badge_r)], outline=(255,226,226), width=1)

# Bottom navigation bar background
nav_top = 2792
draw.rectangle([(0, nav_top), (w, h)], fill=card_bg)
# nav top divider
draw.line([(0, nav_top), (w, nav_top)], fill=nav_top_div, width=1)

# Subtle highlight behind the currently selected nav item area (center-left)
# We do not draw the icon itself, only the pill background to indicate selected state.
selected_center_x = 72  # left-most item in screenshot is highlighted
selected_center_y = nav_top + 52
pill_w = 88
pill_h = 88
draw.ellipse([(selected_center_x - pill_w//2, selected_center_y - pill_h//2),
              (selected_center_x + pill_w//2, selected_center_y + pill_h//2)],
             fill=(255,244,240))

# Small top shadow line above the banner area to anchor content (soft)
shadow_y = banner_y0 - 12
draw.line([(24, shadow_y), (w - 24, shadow_y)], fill=(245,245,245), width=1)

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/00_icon_S94.png
try:
    _c0 = get_crop(0, 462, 519)
    canvas.paste(_c0, (48, 1431), _c0)
except Exception:
    pass
layout["S94+"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/01_icon_Knicks.png
try:
    _c1 = get_crop(1, 1344, 840)
    canvas.paste(_c1, (48, 360), _c1)
except Exception:
    pass
layout["Knicks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/02_icon_August_Wilson_Theatre.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 2183), _c2)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/03_icon_S116.png
try:
    _c3 = get_crop(3, 462, 519)
    canvas.paste(_c3, (546, 1431), _c3)
except Exception:
    pass
layout["S116+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/04_icon_Yankee_Stadium.png
try:
    _c4 = get_crop(4, 1309, 236)
    canvas.paste(_c4, (0, 2419), _c4)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 100, 152)
    canvas.paste(_c5, (1340, 2464), _c5)
except Exception:
    pass
layout["icon_5"] = [1340, 2464, 1440, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/06_icon_View_all.png
try:
    _c6 = get_crop(6, 99, 151)
    canvas.paste(_c6, (1341, 2227), _c6)
except Exception:
    pass
layout["View_all"] = [1341, 2227, 1440, 2378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/07_icon_New_York_NY.png
try:
    _c7 = get_crop(7, 64, 59)
    canvas.paste(_c7, (242, 4), _c7)
except Exception:
    pass
layout["New_York,_NY"] = [242, 4, 306, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/08_icon_888.png
try:
    _c8 = get_crop(8, 144, 240)
    canvas.paste(_c8, (1260, 72), _c8)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/09_icon_6.49_my.png
try:
    _c9 = get_crop(9, 56, 57)
    canvas.paste(_c9, (114, 5), _c9)
except Exception:
    pass
layout["6.49_my"] = [114, 5, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/10_icon_888.png
try:
    _c10 = get_crop(10, 99, 65)
    canvas.paste(_c10, (1214, 0), _c10)
except Exception:
    pass
layout["888"] = [1214, 0, 1313, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/11_icon_6.49_my.png
try:
    _c11 = get_crop(11, 50, 57)
    canvas.paste(_c11, (184, 5), _c11)
except Exception:
    pass
layout["6.49_my"] = [184, 5, 234, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/12_icon_Apr.png
try:
    _c12 = get_crop(12, 264, 183)
    canvas.paste(_c12, (1176, 2000), _c12)
except Exception:
    pass
layout["Apr"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/13_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c13 = get_crop(13, 288, 168)
    canvas.paste(_c13, (864, 2792), _c13)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 64)
    canvas.paste(_c14, (1319, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 1, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 47, 66)
    canvas.paste(_c15, (1154, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1154, 0, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/16_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (576, 2792), _c16)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/17_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (288, 2792), _c17)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 101, 119)
    canvas.paste(_c18, (1339, 2697), _c18)
except Exception:
    pass
layout["icon_18"] = [1339, 2697, 1440, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/19_icon_Browse.png
try:
    _c19 = get_crop(19, 288, 162)
    canvas.paste(_c19, (0, 2792), _c19)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 54, 59)
    canvas.paste(_c21, (316, 5), _c21)
except Exception:
    pass
layout["icon_21"] = [316, 5, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/22_icon_S116.png
try:
    _c22 = get_crop(22, 462, 519)
    canvas.paste(_c22, (546, 1431), _c22)
except Exception:
    pass
layout["S116+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 116, 128)
    canvas.paste(_c23, (1138, 2483), _c23)
except Exception:
    pass
layout["icon_23"] = [1138, 2483, 1254, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/24_icon_New_York_NY.png
try:
    _c24 = get_crop(24, 391, 87)
    canvas.paste(_c24, (39, 119), _c24)
except Exception:
    pass
layout["New_York,_NY"] = [39, 119, 430, 206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/25_text_date.png
try:
    _c25 = get_crop(25, 114, 52)
    canvas.paste(_c25, (137, 208), _c25)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/26_text_Just_for_you.png
try:
    _c26 = get_crop(26, 306, 66)
    canvas.paste(_c26, (38, 1310), _c26)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/27_text_View_all.png
try:
    _c27 = get_crop(27, 264, 183)
    canvas.paste(_c27, (1176, 1248), _c27)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/28_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (576, 2792), _c28)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/29_clickable_Tracking.png
try:
    _c29 = get_crop(29, 72, 72)
    canvas.paste(_c29, (408, 1455), _c29)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_01_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-4/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 72, 72)
    canvas.paste(_c30, (906, 1455), _c30)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
