# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_13
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15.png
# step_index: 13/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile event page
w, h = canvas.size

# Colors (chosen to match the screenshot's dominant tones)
bg_white = (255, 255, 255)
status_bar_bg = (242, 242, 245)      # subtle light grey for status bar
banner_yellow = (255, 200, 0)        # bright yellow banner color
banner_shadow = (240, 170, 0)        # darker yellow for bottom shadow of banner
card_bg = (249, 249, 251)            # very light card background
muted_card = (245, 246, 250)         # slightly different light tone
divider = (235, 233, 238)            # subtle divider line
bottom_bar_bg = (250, 248, 246)      # pale bottom bar
soft_shadow = (0, 0, 0, 30)          # (not used directly - keep for reference)

# 1) Fill whole canvas with the primary background (white)
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# 2) Status bar area (approximately 88px high)
status_bar_h = 88
draw.rectangle([(0, 0), (w, status_bar_h)], fill=status_bar_bg)

# 3) Top banner area (image background)
# Banner spans from just below status bar to around 480px (approx)
banner_top = status_bar_h
banner_bottom = 480
draw.rectangle([(0, banner_top), (w, banner_bottom)], fill=banner_yellow)

# Add a subtle darker band near the bottom of the banner for depth
band_h = 44
draw.rectangle([(0, banner_bottom - band_h), (w, banner_bottom)], fill=banner_shadow)

# Add a thin separator under the banner
draw.line([(48, banner_bottom + 12), (w - 48, banner_bottom + 12)], fill=divider, width=1)

# 4) Organizer card (rounded rectangle) - sits under the title area
organizer_card_left = 48
organizer_card_right = w - 48
organizer_card_top = 980
organizer_card_height = 160
organizer_card_bottom = organizer_card_top + organizer_card_height
draw.rounded_rectangle(
    [(organizer_card_left, organizer_card_top), (organizer_card_right, organizer_card_bottom)],
    radius=20,
    fill=card_bg,
    outline=None
)

# Add subtle inner top highlight (very faint)
draw.line(
    [(organizer_card_left + 2, organizer_card_top + 2), (organizer_card_right - 2, organizer_card_top + 2)],
    fill=(255, 255, 255),
    width=1
)

# 5) Thin divider between main info sections
sep_y1 = organizer_card_bottom + 60
draw.line([(48, sep_y1), (w - 48, sep_y1)], fill=divider, width=1)

# 6) "About this event" area separator (a subtle horizontal rule)
about_sep_y = 1680
draw.line([(48, about_sep_y), (w - 48, about_sep_y)], fill=divider, width=1)

# 7) Location section divider and background cue (subtle)
location_section_top = 2320
draw.rectangle([(0, location_section_top - 12), (w, location_section_top + 4)], fill=bg_white)  # keep white but make area crisp
draw.line([(48, location_section_top), (w - 48, location_section_top)], fill=divider, width=1)

# 8) Bottom ticket bar background with rounded top corners
bottom_bar_top = 2720
draw.rounded_rectangle(
    [(0, bottom_bar_top), (w, h)],
    radius=20,
    fill=bottom_bar_bg,
    outline=None
)

# Add a very faint top divider to separate content and bottom bar
draw.line([(24, bottom_bar_top + 2), (w - 24, bottom_bar_top + 2)], fill=divider, width=1)

# 9) Price area background hint (left side inside bottom bar)
price_hint_left = 24
price_hint_right = 420
price_hint_top = bottom_bar_top + 24
price_hint_bottom = bottom_bar_top + 140
draw.rectangle([(price_hint_left, price_hint_top), (price_hint_right, price_hint_bottom)], fill=bg_white, outline=None)

# 10) Decorative subtle shadows and separators for section grouping
# Slight shadow line under organizer card
draw.line([(organizer_card_left + 6, organizer_card_bottom + 2), (organizer_card_right - 6, organizer_card_bottom + 2)], fill=divider, width=1)

# A few additional subtle dividers where the UI shows section breaks
for y in (1260, 1540, 2040, 2360):
    draw.line([(48, y), (w - 48, y)], fill=divider, width=1)

# Note: All actual icons, buttons, and text are intentionally NOT drawn here.
# This code only provides the background, cards, banners, and separators.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1068), _c0)
except Exception:
    pass
layout["Following"] = [946, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/02_icon_Food_Drink.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2205), _c2)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/03_icon_6_30PM-9_30PA.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["6:30PM-9:30PA"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 67)
    canvas.paste(_c5, (1155, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [1155, 2, 1203, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/06_icon_9.46.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["9.46"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 63)
    canvas.paste(_c7, (1327, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1327, 3, 1372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/08_icon_9.46.png
try:
    _c8 = get_crop(8, 51, 61)
    canvas.paste(_c8, (184, 2), _c8)
except Exception:
    pass
layout["9.46"] = [184, 2, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/09_icon_6.30_PM.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1116, 108), _c9)
except Exception:
    pass
layout["6.30_PM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 56)
    canvas.paste(_c10, (316, 7), _c10)
except Exception:
    pass
layout["icon_10"] = [316, 7, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 100, 64)
    canvas.paste(_c11, (1214, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1214, 2, 1314, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 55, 60)
    canvas.paste(_c12, (247, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 4, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/13_icon_Show_map.png
try:
    _c13 = get_crop(13, 226, 144)
    canvas.paste(_c13, (1166, 2423), _c13)
except Exception:
    pass
layout["Show_map"] = [1166, 2423, 1392, 2567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/14_icon_Join_us_for_this_charitable_event_with_f.png
try:
    _c14 = get_crop(14, 234, 144)
    canvas.paste(_c14, (48, 2205), _c14)
except Exception:
    pass
layout["Join_us_for_this_charitab"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/15_text_9.46.png
try:
    _c15 = get_crop(15, 94, 43)
    canvas.paste(_c15, (20, 15), _c15)
except Exception:
    pass
layout["9.46"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/16_text_The_Stone_House.png
try:
    _c16 = get_crop(16, 364, 144)
    canvas.paste(_c16, (144, 1028), _c16)
except Exception:
    pass
layout["The_Stone_House"] = [144, 1028, 508, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/17_text_88_Followers.png
try:
    _c17 = get_crop(17, 364, 144)
    canvas.paste(_c17, (144, 1028), _c17)
except Exception:
    pass
layout["88_Followers"] = [144, 1028, 508, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/18_text_The_Stone_House_at_Clove_Lakes.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1295), _c18)
except Exception:
    pass
layout["The_Stone_House_at_Clove_"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/19_text_3_hrs.png
try:
    _c19 = get_crop(19, 114, 50)
    canvas.paste(_c19, (139, 1452), _c19)
except Exception:
    pass
layout["3_hrs"] = [139, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/20_text_Refund_policy.png
try:
    _c20 = get_crop(20, 299, 63)
    canvas.paste(_c20, (138, 1558), _c20)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1295), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/22_text_About_this_event.png
try:
    _c22 = get_crop(22, 454, 61)
    canvas.paste(_c22, (45, 1858), _c22)
except Exception:
    pass
layout["About_this_event"] = [45, 1858, 499, 1919]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/23_text_Location.png
try:
    _c23 = get_crop(23, 241, 56)
    canvas.paste(_c23, (42, 2470), _c23)
except Exception:
    pass
layout["Location"] = [42, 2470, 283, 2526]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/24_text_The_Stone_House_at_Clove_Lakes.png
try:
    _c24 = get_crop(24, 234, 144)
    canvas.paste(_c24, (48, 2205), _c24)
except Exception:
    pass
layout["The_Stone_House_at_Clove_"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/25_text_The_Stone_House_at_Clove_Lakes_1150_Clov.png
try:
    _c25 = get_crop(25, 570, 144)
    canvas.paste(_c25, (822, 2768), _c25)
except Exception:
    pass
layout["The_Stone_House_at_Clove_"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_13_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-15/26_text_69.png
try:
    _c26 = get_crop(26, 101, 61)
    canvas.paste(_c26, (89, 2811), _c26)
except Exception:
    pass
layout["$69"] = [89, 2811, 190, 2872]
