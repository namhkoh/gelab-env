# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_09
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11.png
# step_index: 9/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for Eventbrite-like page
# Uses provided canvas (PIL Image) and draw (ImageDraw)

# Colors
BG_WHITE = (255, 255, 255)
STATUS_BAR = (158, 158, 158)       # top status bar grey
HEADER_DIVIDER = (224, 224, 224)   # thin dividers
CARD_OUTLINE = (232, 233, 238)     # card outline / subtle shadow
CARD_BG = (255, 255, 255)
BOTTOM_BAR_BG = (255, 255, 255)
SEPARATOR = (236, 236, 240)
SEARCH_BG = (250, 250, 252)

W, H = canvas.size

def rr(rect, r):
    # small helper to unpack rect
    return (rect[0], rect[1], rect[2], rect[3], r)

# 1) Background fill (ensures consistent base)
draw.rectangle((0, 0, W, H), fill=BG_WHITE)

# 2) Status bar area (top)
status_h = 96
draw.rectangle((0, 0, W, status_h), fill=STATUS_BAR)

# subtle separation line under status bar
draw.line((0, status_h-1, W, status_h-1), fill=HEADER_DIVIDER)

# 3) Header / toolbar area
header_top = status_h
header_bottom = 220
draw.rectangle((0, header_top, W, header_bottom), fill=BG_WHITE)
# light horizontal divider under header
draw.line((48, header_bottom, W-48, header_bottom), fill=HEADER_DIVIDER, width=2)

# 4) Light search bar background inside header (do not draw icons/text)
search_left = 48
search_top = header_top + 20
search_right = W - 48
search_bottom = header_top + 96
draw.rounded_rectangle((search_left, search_top, search_right, search_bottom),
                       radius=28, fill=SEARCH_BG, outline=SEPARATOR, width=1)

# 5) Thin separator between header and content
sep_y = header_bottom + 18
draw.line((36, sep_y, W-36, sep_y), fill=SEPARATOR, width=1)

# 6) First event card container (rounded card background + subtle outline)
# The detected first event image sits at (48,676) size (1344x1194).
# Draw a card container slightly larger behind it.
card1_left = 36
card1_top = 660
card1_right = card1_left + 1368  # 36 + 1368 = 1404
card1_bottom = 1890
draw.rounded_rectangle((card1_left, card1_top, card1_right, card1_bottom),
                       radius=28, fill=CARD_BG, outline=CARD_OUTLINE, width=2)

# subtle top highlight on card
draw.line((card1_left+8, card1_top+6, card1_right-8, card1_top+6), fill=(245,245,247), width=2)

# 7) Second event card container
# Detected second image at (48,1918) size (1344x898).
card2_left = 36
card2_top = 1884
card2_right = card2_left + 1368
card2_bottom = 2852
draw.rounded_rectangle((card2_left, card2_top, card2_right, card2_bottom),
                       radius=28, fill=CARD_BG, outline=CARD_OUTLINE, width=2)

draw.line((card2_left+8, card2_top+6, card2_right-8, card2_top+6), fill=(245,245,247), width=2)

# 8) Section separators between cards and below content areas
# A subtle divider between card sections (above bottom navigation)
draw.line((36, card2_bottom + 12, W-36, card2_bottom + 12), fill=HEADER_DIVIDER, width=1)

# 9) Bottom navigation bar area
bottom_bar_h = 156  # height similar to screenshot bottom controls area
bottom_top = H - bottom_bar_h
draw.rectangle((0, bottom_top, W, H), fill=BOTTOM_BAR_BG)
# top border for bottom nav
draw.line((24, bottom_top, W-24, bottom_top), fill=HEADER_DIVIDER, width=2)

# 10) Small floating separators and subtle shadows near top of card list
# faint horizontal rule under "10,000 events" area (do not draw text)
events_header_rule_y = 620
draw.line((36, events_header_rule_y, W-36, events_header_rule_y), fill=SEPARATOR, width=1)

# 11) Left edge app vertical guideline (very subtle)
draw.line((36, 0, 36, H), fill=(250,250,250), width=1)

# 12) Decorative rounded container behind filter chips row (only background, not chips)
# Chips row is at around y ~ 400; draw a faint horizontal band to anchor chips
chips_band_top = 360
chips_band_bottom = 460
draw.rectangle((36, chips_band_top, W-36, chips_band_bottom), fill=BG_WHITE)
# subtle inner divider for the chips area
draw.line((36, chips_band_bottom, W-36, chips_band_bottom), fill=SEPARATOR, width=1)

# End of structural drawing. The actual icons, images and texts will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2434), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/05_icon_March_22.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["March_22_&"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/06_icon_Foo.png
try:
    _c6 = get_crop(6, 148, 110)
    canvas.paste(_c6, (1283, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2434), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/08_icon_March_22.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["March_22_&"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 59, 62)
    canvas.paste(_c9, (245, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [245, 1, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (1151, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 0, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/11_icon_Foo.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/12_icon_9.45.png
try:
    _c12 = get_crop(12, 124, 116)
    canvas.paste(_c12, (56, 113), _c12)
except Exception:
    pass
layout["9.45"] = [56, 113, 180, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/13_icon_Ramadan_Muslim_Lights_Festival.png
try:
    _c13 = get_crop(13, 1344, 1194)
    canvas.paste(_c13, (48, 676), _c13)
except Exception:
    pass
layout["Ramadan_Muslim_Lights_Fes"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 100, 63)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 59, 63)
    canvas.paste(_c15, (312, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [312, 1, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/16_icon_9.45.png
try:
    _c16 = get_crop(16, 56, 63)
    canvas.paste(_c16, (182, 0), _c16)
except Exception:
    pass
layout["9.45"] = [182, 0, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 56, 62)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/18_icon_New_York.png
try:
    _c18 = get_crop(18, 434, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/19_icon_7_00_PM_EDT.png
try:
    _c19 = get_crop(19, 1344, 1194)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["7:00_PM_EDT"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/20_icon_9.45.png
try:
    _c20 = get_crop(20, 57, 65)
    canvas.paste(_c20, (114, 0), _c20)
except Exception:
    pass
layout["9.45"] = [114, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/21_icon_Ticket_sales_end_soon.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/22_icon_Food_Drink.png
try:
    _c22 = get_crop(22, 1344, 191)
    canvas.paste(_c22, (48, 72), _c22)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/23_icon_Food_Drink.png
try:
    _c23 = get_crop(23, 49, 62)
    canvas.paste(_c23, (383, 2), _c23)
except Exception:
    pass
layout["Food_&_Drink"] = [383, 2, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/24_icon_Ticket_sales_end_soon.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/26_icon_The_HOLI-CON_Ball_Festival_230_Sth_Rooft.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["The_HOLI-CON_Ball_Festiva"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/27_icon_SPECUAL_CUEST_DJ.png
try:
    _c27 = get_crop(27, 1344, 898)
    canvas.paste(_c27, (48, 1918), _c27)
except Exception:
    pass
layout["SPECUAL_CUEST_DJ"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/28_icon_Promoted.png
try:
    _c28 = get_crop(28, 243, 61)
    canvas.paste(_c28, (85, 1765), _c28)
except Exception:
    pass
layout["Promoted"] = [85, 1765, 328, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/29_icon_The_HOLI-CON_Ball_Festival_230_Sth_Rooft.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (864, 2804), _c29)
except Exception:
    pass
layout["The_HOLI-CON_Ball_Festiva"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/30_text_9.45.png
try:
    _c30 = get_crop(30, 94, 43)
    canvas.paste(_c30, (20, 15), _c30)
except Exception:
    pass
layout["9.45"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/31_text_10_000_events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/32_text_NY_USA.png
try:
    _c32 = get_crop(32, 159, 52)
    canvas.paste(_c32, (1170, 1704), _c32)
except Exception:
    pass
layout["NY,_USA"] = [1170, 1704, 1329, 1756]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/33_text_230_FIFTH_ROOFTOP.png
try:
    _c33 = get_crop(33, 525, 96)
    canvas.paste(_c33, (288, 1919), _c33)
except Exception:
    pass
layout["230_FIFTH_ROOFTOP"] = [288, 1919, 813, 2015]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_09_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-11/34_text_SATurday_MARCH_23rd.png
try:
    _c34 = get_crop(34, 1344, 898)
    canvas.paste(_c34, (48, 1918), _c34)
except Exception:
    pass
layout["SATurday_MARCH_23rd"] = [48, 1918, 1392, 2816]
