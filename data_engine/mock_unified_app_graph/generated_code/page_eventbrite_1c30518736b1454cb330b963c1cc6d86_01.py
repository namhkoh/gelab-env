# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_01
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3.png
# step_index: 1/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for Eventbrite-like list page
# Available objects: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (255, 255, 255)            # page background (dominant white)
status_bar_color = (190, 190, 190)    # light grey status bar
header_bg = (255, 255, 255)           # header/toolbar background (white)
divider_color = (230, 230, 230)       # subtle divider
card_bg = (255, 255, 255)             # card background (white)
thumb_bg = (243, 246, 249)            # thumbnail placeholder background (very light)
muted_overlay = (245, 245, 246)       # subtle band / banner backgrounds
bottom_nav_bg = (255, 255, 255)       # bottom navigation background
shadow_line = (235, 235, 236)         # faint shadow/separator

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area) ~56px high
status_h = 56
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / toolbar background below status bar (reserve area for logo + search)
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)
# header bottom divider
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=divider_color, width=1)

# Section title area separator (space where "More events you'll love" sits)
# We only draw a faint underline to separate header from content
section_sep_y = 460
draw.line([(48, section_sep_y), (w-48, section_sep_y)], fill=divider_color, width=1)

# Define card bounds (use detected event card vertical positions)
cards = [
    (48,  886, 1392, 1282),  # first visible event card
    (48, 1282, 1392, 1678),  # second event card
    (48, 1678, 1392, 2074),  # third
    (48, 2074, 1392, 2470),  # fourth
    (48, 2470, 1392, 2816),  # fifth (near bottom)
]

# Draw each card background and separators
for (x1, y1, x2, y2) in cards:
    # card white rounded rectangle
    radius = 14
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=radius, fill=card_bg, outline=None)
    # subtle top and bottom separators/shadows to give separation
    draw.line([(x1+8, y1), (x2-8, y1)], fill=shadow_line, width=1)
    draw.line([(x1+8, y2), (x2-8, y2)], fill=divider_color, width=1)
    # left thumbnail placeholder area (do not draw icons/text, just neutral background block)
    thumb_w = 160
    thumb_h = 160
    thumb_x = x1 + 0
    # center thumbnail vertically inside card with 16px padding
    thumb_y = y1 + 16
    thumb_box = (thumb_x + 8, thumb_y, thumb_x + 8 + thumb_w, thumb_y + thumb_h)
    draw.rounded_rectangle([thumb_box[0:2], thumb_box[2:4]], radius=8, fill=thumb_bg, outline=None)
    # small badge background area (top-left of thumbnail) as muted rounded rect
    badge_w, badge_h = 96, 44
    badge_x = thumb_box[0] + 8
    badge_y = thumb_box[1] + 8
    draw.rounded_rectangle([(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)], radius=8, fill=muted_overlay)

# Draw horizontal separators between list groups (extra faint lines across content)
separator_ys = [1282, 1678, 2074, 2470, 2804]
for y in separator_ys:
    draw.line([(48, y), (w-48, y)], fill=divider_color, width=1)

# Floating location pill area - we must not draw the pill content itself (auto-pasted).
# Instead draw a very faint backdrop where the pill sits so it reads as structural background.
loc_pill_w = 456
loc_pill_h = 76
loc_pill_x = 492
loc_pill_y = 2596  # slightly above bottom nav area
draw.rounded_rectangle([(loc_pill_x, loc_pill_y), (loc_pill_x + loc_pill_w, loc_pill_y + loc_pill_h)],
                       radius=36, fill=(255,255,255), outline=divider_color)

# Bottom navigation bar background and top divider
bottom_nav_top = 2760
draw.rectangle([(0, bottom_nav_top), (w, h)], fill=bottom_nav_bg)
draw.line([(24, bottom_nav_top), (w-24, bottom_nav_top)], fill=divider_color, width=1)

# Small visual hints for nav items (no icons or text) - simple subtle circular placeholders
nav_centers = [(144, 2880), (432, 2880), (720, 2880), (1008, 2880), (1296, 2880)]
for (cx, cy) in nav_centers:
    r = 28
    draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], fill=(255,255,255), outline=shadow_line)

# Final faint vertical padding lines on the left/right edges to match layout margins
draw.line([(48, header_bottom+8), (48, h-200)], fill=shadow_line, width=1)
draw.line([(w-48, header_bottom+8), (w-48, h-200)], fill=shadow_line, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/00_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 886), _c0)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/01_icon_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/02_icon_FRIDAY.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["FRIDAY"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/03_icon_NDIE.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["NDIE"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/04_icon_NDIE.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/05_icon_Iaightsinel.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1678), _c5)
except Exception:
    pass
layout["Iaightsinel"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/06_icon_9_00_PM_PDT.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 490), _c6)
except Exception:
    pass
layout["9:00_PM_PDT"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/07_icon_Los_Angeles.png
try:
    _c7 = get_crop(7, 456, 117)
    canvas.paste(_c7, (492, 2651), _c7)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 747), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 1951), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/10_icon_Afliccion_Perdida_y.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1140, 1539), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/13_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 1282), _c13)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/14_icon_Sylmai.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (288, 2804), _c14)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 2347), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 1951), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/17_icon_8_21125_creator_followers.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1678), _c17)
except Exception:
    pass
layout["8_21125_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 123)
    canvas.paste(_c18, (1140, 1159), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/19_icon_Overflow_menu_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1284, 1159), _c19)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/20_icon_Overflow_menu_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1284, 1539), _c20)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 59)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/23_icon_4.53.png
try:
    _c23 = get_crop(23, 99, 95)
    canvas.paste(_c23, (43, 123), _c23)
except Exception:
    pass
layout["4.53"] = [43, 123, 142, 218]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/24_icon_4.53.png
try:
    _c24 = get_crop(24, 56, 61)
    canvas.paste(_c24, (182, 2), _c24)
except Exception:
    pass
layout["4.53"] = [182, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 50, 60)
    canvas.paste(_c25, (248, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 2, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 47, 53)
    canvas.paste(_c26, (1321, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/27_icon_Public_House_Los_Angeles_CA.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 886), _c27)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/28_icon_4.53.png
try:
    _c28 = get_crop(28, 59, 61)
    canvas.paste(_c28, (115, 2), _c28)
except Exception:
    pass
layout["4.53"] = [115, 2, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 66, 57)
    canvas.paste(_c29, (1212, 5), _c29)
except Exception:
    pass
layout["icon_29"] = [1212, 5, 1278, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/30_icon_Free.png
try:
    _c30 = get_crop(30, 1344, 346)
    canvas.paste(_c30, (48, 2470), _c30)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 56)
    canvas.paste(_c31, (385, 7), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/32_icon_Free.png
try:
    _c32 = get_crop(32, 126, 74)
    canvas.paste(_c32, (247, 560), _c32)
except Exception:
    pass
layout["Free"] = [247, 560, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 41, 55)
    canvas.paste(_c33, (1272, 6), _c33)
except Exception:
    pass
layout["icon_33"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/34_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1282), _c34)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/35_icon_Punk_Indie_Rock_Dance_Party.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 2074), _c35)
except Exception:
    pass
layout["Punk;_Indie_Rock_Dance_Pa"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/36_icon_Tickets.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/37_icon_BIZARRE_LOVE_TRIANGLE_New_Wave_Post.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["BIZARRE_LOVE_TRIANGLE:_Ne"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/38_icon_31_creator_followers.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (576, 2804), _c38)
except Exception:
    pass
layout["31_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/39_icon_5.30_PM_PDT.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/40_text_4.53.png
try:
    _c40 = get_crop(40, 89, 43)
    canvas.paste(_c40, (22, 17), _c40)
except Exception:
    pass
layout["4.53"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/42_text_Mon_May_13.png
try:
    _c42 = get_crop(42, 222, 43)
    canvas.paste(_c42, (393, 2525), _c42)
except Exception:
    pass
layout["Mon,_May_13"] = [393, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/43_text_5.30_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/44_text_31_creator_followers.png
try:
    _c44 = get_crop(44, 1344, 346)
    canvas.paste(_c44, (48, 2470), _c44)
except Exception:
    pass
layout["31_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_01_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
