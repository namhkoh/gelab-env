# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_03
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5.png
# step_index: 3/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile page
# Variables provided: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

width, height = canvas.size

# Common colors
status_bar_color = (189, 189, 189)     # light gray status bar
header_underline_color = (43, 102, 255)  # bright blue underline used in header
section_card_fill = (250, 251, 253)    # very subtle off-white / bluish card background
section_card_border = (230, 233, 240)  # light border for cards
page_bg = (255, 255, 255)              # main page background (white)
separator_color = (233, 236, 240)      # thin separators
nav_bg = (255, 255, 255)               # bottom nav background
shadow_line = (220, 222, 225)

# Fill overall background (canvas already white, but set explicitly)
draw.rectangle([0, 0, width, height], fill=page_bg)

# Status bar (top ~56px)
status_h = 56
draw.rectangle([0, 0, width, status_h], fill=status_bar_color)

# Header area below status bar (search/header region)
header_top = status_h
header_h = 80
header_bottom = header_top + header_h
# keep header background white (no content drawn), add a subtle drop shadow line
draw.rectangle([0, header_top, width, header_bottom], fill=page_bg)
draw.line([16, header_bottom, width-16, header_bottom], fill=shadow_line, width=1)

# Blue underline below the header (thin prominent accent)
underline_y = header_bottom + 6
draw.line([48, underline_y, width-48, underline_y], fill=header_underline_color, width=4)

# Section separators and structural divider under search area
divider_y = underline_y + 12
draw.line([24, divider_y, width-24, divider_y], fill=separator_color, width=1)

# Draw rounded "cards" / section backgrounds for listed groups
# Use the detected element boxes (only drawing backgrounds, not content)
cards = [
    # (x0, y0, x1, y1)
    (48, 378, 48+1344, 378+120),   # tech conference card
    (48, 618, 48+1344, 618+120),   # tech meetup card
    (48, 738, 48+1344, 738+120),   # techno music card
    (48, 858, 48+1344, 858+144),   # tech events card / popular cluster
    (48, 1117, 48+1344, 1117+396), # events area large card block
    (48, 1513, 48+1344, 1513+396), # another events row block
    (48, 1909, 48+1344, 1909+396), # event listing card
    (48, 2305, 48+1344, 2305+396)  # event listing card further down
]

card_radius = 10
for (x0, y0, x1, y1) in cards:
    # Slightly inset shadow top line for separation
    draw.line([x0+8, y0, x1-8, y0], fill=separator_color, width=1)
    # Draw rounded rectangle as subtle card background
    draw.rounded_rectangle([x0, y0, x1, y1],
                           radius=card_radius,
                           fill=section_card_fill,
                           outline=section_card_border,
                           width=1)

    # Thin separator at bottom of card (subtle)
    draw.line([x0+8, y1, x1-8, y1], fill=separator_color, width=1)

# Add thin separators between event rows inside the big card blocks
# (For the large 396px height blocks, draw a faint divider about 120px from top to emulate stacked rows)
for (x0, y0, x1, y1) in [(48, 1117, 1392, 1117+396), (48, 1513, 1392, 1513+396), (48, 1909, 1392, 1909+396), (48, 2305, 1392, 2305+396)]:
    # draw one or two horizontal separators inside to suggest multiple list items
    row_h = 120
    current_y = y0 + row_h
    while current_y < y1 - 12:
        draw.line([x0+16, current_y, x1-16, current_y], fill=separator_color, width=1)
        current_y += row_h

# Left-side "Popular" list area: draw a subtle left rail background to group the list
popular_group_top = 280
popular_group_bottom = 360
draw.rounded_rectangle([32, popular_group_top, width-32, popular_group_bottom],
                       radius=8, fill=(255,255,255,0), outline=None)

# Events section header area background hint (no text)
events_header_y = 320
draw.rectangle([32, events_header_y, width-32, events_header_y+48], fill=page_bg)
draw.line([32, events_header_y+48, width-32, events_header_y+48], fill=separator_color, width=1)

# Bottom navigation bar background (reserve area for icons that will be pasted)
nav_top = 2804
draw.rectangle([0, nav_top, width, height], fill=nav_bg)
# subtle top border/shadow for nav bar
draw.line([0, nav_top, width, nav_top], fill=separator_color, width=1)
draw.line([0, nav_top+2, width, nav_top+2], fill=shadow_line, width=1)

# Final subtle horizontal separators across full width for structure rhythm
for y in [divider_y + 120, 520, 720, 920, 1120, 1520, 1920, 2320]:
    if 0 < y < height - 200:
        draw.line([24, y, width-24, y], fill=(248,249,250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/00_icon_8_214_creator_followers.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1909), _c0)
except Exception:
    pass
layout["8_214_creator_followers"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/01_icon_5.24.png
try:
    _c1 = get_crop(1, 56, 62)
    canvas.paste(_c1, (115, 2), _c1)
except Exception:
    pass
layout["5.24"] = [115, 2, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/02_icon_1I_O0_PM_EDT.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1513), _c2)
except Exception:
    pass
layout["1I:O0_PM_EDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/03_icon_Tech.png
try:
    _c3 = get_crop(3, 56, 60)
    canvas.paste(_c3, (313, 3), _c3)
except Exception:
    pass
layout["Tech"] = [313, 3, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/04_icon_5.24.png
try:
    _c4 = get_crop(4, 53, 61)
    canvas.paste(_c4, (183, 2), _c4)
except Exception:
    pass
layout["5.24"] = [183, 2, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/05_icon_Online.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1117), _c5)
except Exception:
    pass
layout["Online"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/06_icon_12_00AM_EDT.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1117), _c6)
except Exception:
    pass
layout["12:00AM_EDT"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 41, 55)
    canvas.paste(_c7, (254, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [254, 6, 295, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/08_icon_Fr.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (864, 2804), _c8)
except Exception:
    pass
layout["Fr"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/09_icon_5.24.png
try:
    _c9 = get_crop(9, 122, 104)
    canvas.paste(_c9, (55, 119), _c9)
except Exception:
    pass
layout["5.24"] = [55, 119, 177, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/10_icon_Online.png
try:
    _c10 = get_crop(10, 112, 50)
    canvas.paste(_c10, (390, 2116), _c10)
except Exception:
    pass
layout["Online"] = [390, 2116, 502, 2166]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/11_icon_tech_events.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 858), _c11)
except Exception:
    pass
layout["tech_events"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/12_icon_Online.png
try:
    _c12 = get_crop(12, 112, 50)
    canvas.paste(_c12, (391, 1353), _c12)
except Exception:
    pass
layout["Online"] = [391, 1353, 503, 1403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/13_icon_430_creator_followers.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 2305), _c13)
except Exception:
    pass
layout["430_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/14_icon_5.24.png
try:
    _c14 = get_crop(14, 92, 61)
    canvas.paste(_c14, (16, 2), _c14)
except Exception:
    pass
layout["5.24"] = [16, 2, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/15_icon_techno_music.png
try:
    _c15 = get_crop(15, 1344, 120)
    canvas.paste(_c15, (48, 738), _c15)
except Exception:
    pass
layout["techno_music"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/16_icon_Online.png
try:
    _c16 = get_crop(16, 112, 51)
    canvas.paste(_c16, (391, 1749), _c16)
except Exception:
    pass
layout["Online"] = [391, 1749, 503, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 47, 60)
    canvas.paste(_c17, (1322, 2), _c17)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/18_icon_Joinon.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Joinon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 90, 63)
    canvas.paste(_c19, (1216, 0), _c19)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1306, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1099, 96), _c20)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/21_icon_Tech_Futures_The_Future_of_Tech_Jobs.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1909), _c21)
except Exception:
    pass
layout["Tech_Futures:_The_Future_"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/22_icon_Tech.png
try:
    _c22 = get_crop(22, 181, 102)
    canvas.paste(_c22, (187, 118), _c22)
except Exception:
    pass
layout["Tech"] = [187, 118, 368, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/23_icon_Online.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1513), _c23)
except Exception:
    pass
layout["Online"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/24_icon_women_in_tech.png
try:
    _c24 = get_crop(24, 94, 95)
    canvas.paste(_c24, (32, 529), _c24)
except Exception:
    pass
layout["women_in_tech"] = [32, 529, 126, 624]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 149, 144)
    canvas.paste(_c25, (1243, 97), _c25)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/26_icon_techno_music.png
try:
    _c26 = get_crop(26, 92, 97)
    canvas.paste(_c26, (34, 766), _c26)
except Exception:
    pass
layout["techno_music"] = [34, 766, 126, 863]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/28_icon_tech_conference.png
try:
    _c28 = get_crop(28, 1344, 120)
    canvas.paste(_c28, (48, 378), _c28)
except Exception:
    pass
layout["tech_conference"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/29_icon_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/30_icon_Online.png
try:
    _c30 = get_crop(30, 112, 50)
    canvas.paste(_c30, (390, 2512), _c30)
except Exception:
    pass
layout["Online"] = [390, 2512, 502, 2562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/31_icon_Events.png
try:
    _c31 = get_crop(31, 90, 88)
    canvas.paste(_c31, (35, 890), _c31)
except Exception:
    pass
layout["Events"] = [35, 890, 125, 978]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/32_text_Popular.png
try:
    _c32 = get_crop(32, 221, 78)
    canvas.paste(_c32, (44, 298), _c32)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/33_text_women_in_tech.png
try:
    _c33 = get_crop(33, 275, 43)
    canvas.paste(_c33, (165, 554), _c33)
except Exception:
    pass
layout["women_in_tech"] = [165, 554, 440, 597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/34_text_tech_meetup.png
try:
    _c34 = get_crop(34, 1344, 120)
    canvas.paste(_c34, (48, 618), _c34)
except Exception:
    pass
layout["tech_meetup"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/35_text_Events.png
try:
    _c35 = get_crop(35, 191, 61)
    canvas.paste(_c35, (45, 1026), _c35)
except Exception:
    pass
layout["Events"] = [45, 1026, 236, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/36_text_Fr.png
try:
    _c36 = get_crop(36, 42, 14)
    canvas.paste(_c36, (782, 2794), _c36)
except Exception:
    pass
layout["Fr"] = [782, 2794, 824, 2808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/37_clickable_Tech.png
try:
    _c37 = get_crop(37, 1344, 191)
    canvas.paste(_c37, (48, 72), _c37)
except Exception:
    pass
layout["Tech"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/38_clickable_women_in_tech.png
try:
    _c38 = get_crop(38, 1344, 120)
    canvas.paste(_c38, (48, 498), _c38)
except Exception:
    pass
layout["women_in_tech"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_03_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-5/39_clickable_Favorites.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (576, 2804), _c39)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
