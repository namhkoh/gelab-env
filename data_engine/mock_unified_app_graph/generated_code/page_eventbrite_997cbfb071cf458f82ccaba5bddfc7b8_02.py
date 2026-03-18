# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_02
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4.png
# step_index: 2/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall page background (very light off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# Status bar (top area, neutral gray)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")

# Subtle bottom edge for status bar to separate from content
draw.line([(0, status_h), (1440, status_h)], fill="#BDBDBD", width=1)

# Search bar background (rounded) beneath the status bar
search_x0, search_y0 = 48, 72
search_x1, search_y1 = 1392, 72 + 190  # match approximate detected search area height
draw.rounded_rectangle(
    [(search_x0 - 8, search_y0 + 8), (search_x1 + 8, search_y1 + 8)],
    radius=36,
    fill="#F2F7FB",
    outline=None
)

# Thin divider under the search area
divider_y = search_y1 + 8 + 8
draw.line([(48, divider_y), (1392, divider_y)], fill="#E3E5E8", width=2)

# Subtle horizontal spacing area (e.g., location row separation)
loc_row_y = divider_y + 12
draw.line([(48, loc_row_y), (1392, loc_row_y)], fill="#F0F1F3", width=1)

# Main event card group background (rounded card area)
# This acts as a background panel behind the top event content (keeps content clear)
card1_x0, card1_y0 = 40, 660
card1_x1, card1_y1 = 1400, 1878  # tall area to cover the first big event area
# shadow (subtle)
draw.rounded_rectangle(
    [(card1_x0 + 6, card1_y0 + 8), (card1_x1 + 6, card1_y1 + 8)],
    radius=20,
    fill="#EFEFF1",
    outline=None
)
# card surface
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=20,
    fill="#FFFFFF",
    outline="#E6E6EA",
    width=1
)

# Small separator to mark the label area above the card (e.g., "10,000 events" zone)
draw.line([(48, 548), (1392, 548)], fill="#E9EAED", width=1)

# Section divider before the next event area
section_sep_y = card1_y1 + 24
draw.line([(48, section_sep_y), (1392, section_sep_y)], fill="#E6E7EA", width=2)

# Second event/image post background card (rounded)
card2_x0, card2_y0 = 40, 1830
card2_x1, card2_y1 = 1400, 2820
# shadow for elevation
draw.rounded_rectangle(
    [(card2_x0 + 6, card2_y0 + 8), (card2_x1 + 6, card2_y1 + 8)],
    radius=24,
    fill="#EFEFF1",
    outline=None
)
# card surface
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=24,
    fill="#FFFFFF",
    outline="#E6E6EA",
    width=1
)

# Divider lines between content blocks (subtle)
for y in [card1_y1 + 12, card2_y1 + 8]:
    draw.line([(48, y), (1392, y)], fill="#F1F2F4", width=1)

# Bottom navigation bar background and top divider
nav_h = 160
nav_y0 = 2960 - nav_h
draw.rectangle([(0, nav_y0), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, nav_y0), (1440, nav_y0)], fill="#E2E3E6", width=2)

# Subtle indicator for active area in nav (no icons/text drawn)
active_indicator_w = 96
draw.rounded_rectangle(
    [(720 - active_indicator_w // 2 - 6, nav_y0 + 18), (720 + active_indicator_w // 2 + 6, nav_y0 + 18 + 6)],
    radius=6,
    fill="#FFF7F0",
    outline=None
)

# Final subtle borders and finish touches: light left/right margins line
draw.line([(40, status_h + 8), (40, 2960 - nav_h - 8)], fill="#FAFAFB", width=1)
draw.line([(1400, status_h + 8), (1400, 2960 - nav_h - 8)], fill="#FAFAFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2434), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2434), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/07_icon_Free_Street_Parking.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["Free_Street_Parking"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 57, 61)
    canvas.paste(_c8, (246, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/09_icon_9.15.png
try:
    _c9 = get_crop(9, 123, 112)
    canvas.paste(_c9, (56, 116), _c9)
except Exception:
    pass
layout["9.15"] = [56, 116, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/11_icon_9.15.png
try:
    _c11 = get_crop(11, 53, 62)
    canvas.paste(_c11, (183, 0), _c11)
except Exception:
    pass
layout["9.15"] = [183, 0, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 61, 62)
    canvas.paste(_c12, (311, 1), _c12)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 96, 59)
    canvas.paste(_c13, (1207, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1207, 0, 1303, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 63, 58)
    canvas.paste(_c14, (1316, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1316, 0, 1379, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/15_icon_Los_Angeles.png
try:
    _c15 = get_crop(15, 492, 144)
    canvas.paste(_c15, (0, 259), _c15)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/16_icon_Address_willlbe_given_to_attendees.png
try:
    _c16 = get_crop(16, 1344, 1194)
    canvas.paste(_c16, (48, 676), _c16)
except Exception:
    pass
layout["Address_willlbe_given_to_"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/17_icon_9.15.png
try:
    _c17 = get_crop(17, 55, 64)
    canvas.paste(_c17, (116, 0), _c17)
except Exception:
    pass
layout["9.15"] = [116, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/18_icon_Overflow_menu_button.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1236, 1192), _c18)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 50, 60)
    canvas.paste(_c19, (383, 3), _c19)
except Exception:
    pass
layout["Search_forae"] = [383, 3, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/20_icon_Guided_Flower.png
try:
    _c20 = get_crop(20, 1344, 1194)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Guided_Flower"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/21_icon_LAFW_CELEBRITY_RUNWAY_SHOW.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["LAFW_CELEBRITY_RUNWAY_SHO"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/22_icon_Ticket_sales_end_soon.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/23_icon_2125_N_Buena_Vista_St.png
try:
    _c23 = get_crop(23, 44, 60)
    canvas.paste(_c23, (284, 1765), _c23)
except Exception:
    pass
layout["2125_N_Buena_Vista_St"] = [284, 1765, 328, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/24_icon_LAFW_CELEBRITY_RUNWAY_SHOW.png
try:
    _c24 = get_crop(24, 1344, 898)
    canvas.paste(_c24, (48, 1918), _c24)
except Exception:
    pass
layout["LAFW_CELEBRITY_RUNWAY_SHO"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/25_icon_LAFW_CELEBRITY_RUNWAY_SHOW.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["LAFW_CELEBRITY_RUNWAY_SHO"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/26_icon_Ticket_sales_end_soon.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 39, 60)
    canvas.paste(_c27, (1275, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1275, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/28_text_9.15.png
try:
    _c28 = get_crop(28, 94, 43)
    canvas.paste(_c28, (20, 17), _c28)
except Exception:
    pass
layout["9.15"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/29_text_10_000_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/30_text_2125_N_Buena_Vista_St.png
try:
    _c30 = get_crop(30, 425, 54)
    canvas.paste(_c30, (90, 1704), _c30)
except Exception:
    pass
layout["2125_N_Buena_Vista_St"] = [90, 1704, 515, 1758]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/31_clickable_Tickets.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (864, 2804), _c31)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_02_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-4/32_clickable_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
