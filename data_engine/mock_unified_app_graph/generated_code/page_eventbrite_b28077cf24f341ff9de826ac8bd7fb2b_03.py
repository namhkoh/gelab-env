# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_03
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5.png
# step_index: 3/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for a 1440x2960 canvas.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full canvas background (dominant color from screenshot: white / very light)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top ~56px) - muted gray background
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill=(135, 135, 135))

# Header / toolbar area beneath status bar
header_y0 = status_h
header_y1 = 140
draw.rectangle((0, header_y0, 1440, header_y1), fill=(255, 255, 255))

# Blue underline under the header (matches the accent underline in screenshot)
underline_x0 = 48
underline_x1 = 1392
underline_h = 4
draw.rectangle((underline_x0, header_y1 - underline_h - 2, underline_x1, header_y1 - 2), fill=(13, 75, 210))

# subtle divider under the header
draw.line((0, header_y1, 1440, header_y1), fill=(230, 230, 235), width=1)

# "Popular" group background (rounded card behind the list of popular search chips)
popular_card_x0 = 32
popular_card_x1 = 1408
popular_card_y0 = 260
popular_card_y1 = 740
draw.rounded_rectangle(
    (popular_card_x0, popular_card_y0, popular_card_x1, popular_card_y1),
    radius=12,
    fill=(250, 250, 251),
    outline=(230, 230, 235),
    width=1
)

# Divider between Popular and Events area
events_div_y = 920
draw.line((48, events_div_y, 1392, events_div_y), fill=(240, 240, 242), width=1)

# Event list cards (four event rows). Draw lightweight rounded cards / group backgrounds.
event_card_x0 = 48
event_card_x1 = 1392
event_starts = [1117, 1513, 1909, 2305]
card_height = 360
for y in event_starts:
    top = y
    bottom = y + card_height
    # card background (white with very light border to separate from page)
    draw.rounded_rectangle(
        (event_card_x0, top, event_card_x1, bottom),
        radius=8,
        fill=(255, 255, 255),
        outline=(236, 236, 240),
        width=1
    )
    # subtle inner separator line at bottom of card to enhance separation
    draw.line((event_card_x0 + 12, bottom, event_card_x1 - 12, bottom), fill=(245, 245, 247), width=1)

# Thin separators between event rows (in case cards are flush with background)
for idx in range(len(event_starts) - 1):
    sep_y = event_starts[idx] + card_height + 18
    draw.line((48, sep_y, 1392, sep_y), fill=(248, 248, 249), width=1)

# Bottom navigation bar area (around y=2804 height ~156)
nav_y0 = 2804
nav_y1 = 2960
draw.rectangle((0, nav_y0, 1440, nav_y1), fill=(255, 255, 255))
# top divider for nav
draw.line((0, nav_y0, 1440, nav_y0), fill=(230, 230, 235), width=1)
# slight shadow under nav (very faint)
draw.rectangle((0, nav_y1 - 6, 1440, nav_y1), fill=(252, 252, 252))

# Page left and right outer margins hint (very faint vertical guides to match screenshot spacing)
draw.line((48, 0, 48, 2960), fill=(255, 255, 255), width=0)  # no visible change, placeholder to preserve margin concept
draw.line((1392, 0, 1392, 2960), fill=(255, 255, 255), width=0)

# Final subtle page-wide vertical rhythm divider near content area (very light)
draw.line((48, 200, 1392, 200), fill=(250, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/00_icon_Online.png
try:
    _c0 = get_crop(0, 112, 50)
    canvas.paste(_c0, (390, 2147), _c0)
except Exception:
    pass
layout["Online"] = [390, 2147, 502, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/01_icon_Online.png
try:
    _c1 = get_crop(1, 111, 47)
    canvas.paste(_c1, (390, 1752), _c1)
except Exception:
    pass
layout["Online"] = [390, 1752, 501, 1799]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/02_icon_5_00_PM_EDT.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1513), _c2)
except Exception:
    pass
layout["5:00_PM_EDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/03_icon_5_00_PM_EDT.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1909), _c3)
except Exception:
    pass
layout["5:00_PM_EDT"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/04_icon_Wellness.png
try:
    _c4 = get_crop(4, 57, 60)
    canvas.paste(_c4, (312, 3), _c4)
except Exception:
    pass
layout["Wellness"] = [312, 3, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/05_icon_Tickets.png
try:
    _c5 = get_crop(5, 288, 156)
    canvas.paste(_c5, (864, 2804), _c5)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/06_icon_4.44.png
try:
    _c6 = get_crop(6, 53, 60)
    canvas.paste(_c6, (183, 3), _c6)
except Exception:
    pass
layout["4.44"] = [183, 3, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/07_icon_4.44.png
try:
    _c7 = get_crop(7, 56, 62)
    canvas.paste(_c7, (115, 2), _c7)
except Exception:
    pass
layout["4.44"] = [115, 2, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/08_icon_6_00_PM_PDT.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1117), _c8)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 55)
    canvas.paste(_c9, (253, 6), _c9)
except Exception:
    pass
layout["icon_9"] = [253, 6, 296, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/10_icon_4.44.png
try:
    _c10 = get_crop(10, 117, 106)
    canvas.paste(_c10, (58, 118), _c10)
except Exception:
    pass
layout["4.44"] = [58, 118, 175, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/11_icon_Online.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 2305), _c11)
except Exception:
    pass
layout["Online"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/12_icon_8_119_creator_followers.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["8_119_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/13_icon_More.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (1152, 2804), _c13)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/14_icon_4.44.png
try:
    _c14 = get_crop(14, 90, 61)
    canvas.paste(_c14, (17, 2), _c14)
except Exception:
    pass
layout["4.44"] = [17, 2, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/15_icon_Wellness.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Wellness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/16_icon_Online.png
try:
    _c16 = get_crop(16, 112, 49)
    canvas.paste(_c16, (390, 2513), _c16)
except Exception:
    pass
layout["Online"] = [390, 2513, 502, 2562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/17_icon_8_119_creator_followers.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (576, 2804), _c17)
except Exception:
    pass
layout["8_119_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 92, 63)
    canvas.paste(_c18, (1216, 0), _c18)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1308, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 47, 60)
    canvas.paste(_c19, (1322, 2), _c19)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/20_icon_Online.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1513), _c20)
except Exception:
    pass
layout["Online"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/21_icon_Online.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1909), _c21)
except Exception:
    pass
layout["Online"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/22_icon_Wellness_Wednesday.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 2305), _c22)
except Exception:
    pass
layout["Wellness_Wednesday"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1099, 96), _c23)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/24_icon_Cancel.png
try:
    _c24 = get_crop(24, 149, 144)
    canvas.paste(_c24, (1243, 97), _c24)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/25_icon_Events.png
try:
    _c25 = get_crop(25, 82, 83)
    canvas.paste(_c25, (38, 892), _c25)
except Exception:
    pass
layout["Events"] = [38, 892, 120, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/26_icon_Wellness.png
try:
    _c26 = get_crop(26, 46, 63)
    canvas.paste(_c26, (385, 2), _c26)
except Exception:
    pass
layout["Wellness"] = [385, 2, 431, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/27_icon_bricolage_wellness_holiday_open_house.png
try:
    _c27 = get_crop(27, 1344, 120)
    canvas.paste(_c27, (48, 738), _c27)
except Exception:
    pass
layout["bricolage_wellness_holida"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/28_icon_12_00_PM_EDT.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 2305), _c28)
except Exception:
    pass
layout["12:00_PM_EDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/29_icon_12_00_PM_EDT.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2305), _c29)
except Exception:
    pass
layout["12:00_PM_EDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 94, 93)
    canvas.paste(_c30, (32, 530), _c30)
except Exception:
    pass
layout["icon_30"] = [32, 530, 126, 623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/31_text_Popular.png
try:
    _c31 = get_crop(31, 221, 78)
    canvas.paste(_c31, (44, 298), _c31)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/32_text_health_and_wellness.png
try:
    _c32 = get_crop(32, 1344, 120)
    canvas.paste(_c32, (48, 378), _c32)
except Exception:
    pass
layout["health_and_wellness"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/33_text_wellness_retreat.png
try:
    _c33 = get_crop(33, 1344, 120)
    canvas.paste(_c33, (48, 498), _c33)
except Exception:
    pass
layout["wellness_retreat"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/34_text_black_wellness.png
try:
    _c34 = get_crop(34, 1344, 120)
    canvas.paste(_c34, (48, 618), _c34)
except Exception:
    pass
layout["black_wellness"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/35_text_mental_wellness.png
try:
    _c35 = get_crop(35, 302, 43)
    canvas.paste(_c35, (168, 912), _c35)
except Exception:
    pass
layout["mental_wellness"] = [168, 912, 470, 955]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/36_text_Events.png
try:
    _c36 = get_crop(36, 188, 61)
    canvas.paste(_c36, (45, 1026), _c36)
except Exception:
    pass
layout["Events"] = [45, 1026, 233, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/37_clickable_mental_wellness.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 858), _c37)
except Exception:
    pass
layout["mental_wellness"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_03_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-5/38_clickable_Home.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (0, 2804), _c38)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
