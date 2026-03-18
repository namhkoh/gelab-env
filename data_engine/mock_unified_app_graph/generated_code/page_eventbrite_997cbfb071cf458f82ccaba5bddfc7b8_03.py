# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_03
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5.png
# step_index: 3/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile page
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw)

# Full background (page white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top area)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(189, 189, 189))
# subtle bottom border under status bar
draw.line((0, status_h - 1, 1440, status_h - 1), fill=(170, 170, 170), width=1)

# Light horizontal band behind the main heading area (subtle tint)
draw.rectangle((0, 200, 1440, 260), fill=(250, 249, 251))

# Primary heading divider (thin line below "Find events in..." heading)
heading_div_y = 340
draw.line((48, heading_div_y, 1392, heading_div_y), fill=(220, 213, 226), width=2)

# Secondary subtle separator (above the main divider, for visual layering)
draw.line((48, heading_div_y - 24, 1392, heading_div_y - 24), fill=(245, 243, 246), width=1)

# Separator between options area and the location list (light)
draw.line((48, 560, 1392, 560), fill=(245, 243, 246), width=1)

# Card / section background for "Browsing in / Los Angeles" area
card_x0, card_y0 = 48, 760
card_x1, card_y1 = 1392, 920
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1),
                       radius=20,
                       fill=(255, 255, 255),
                       outline=(238, 232, 241),
                       width=1)

# Subtle shadow line beneath the card to lift it slightly
draw.rectangle((card_x0, card_y1, card_x1, card_y1 + 4), fill=(248, 246, 249))

# Right-side subtle panel (keeps right area visually balanced without drawing icons)
draw.rectangle((1220, 200, 1440, 920), fill=(255, 255, 255))

# Left margin guide (very faint to echo layout spacing)
draw.line((48, status_h, 48, 2960), fill=(255, 255, 255), width=1)

# Bottom-of-header subtle horizontal rule to separate toolbar region (very faint)
draw.line((48, 180, 1392, 180), fill=(250, 249, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/01_icon_9.15.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["9.15"] = [180, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/02_icon_9.15.png
try:
    _c2 = get_crop(2, 168, 168)
    canvas.paste(_c2, (0, 72), _c2)
except Exception:
    pass
layout["9.15"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/03_icon_9.15.png
try:
    _c3 = get_crop(3, 50, 63)
    canvas.paste(_c3, (118, 2), _c3)
except Exception:
    pass
layout["9.15"] = [118, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 62)
    canvas.paste(_c4, (315, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 2, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 62, 63)
    canvas.paste(_c5, (1212, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1212, 1, 1274, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 58)
    canvas.paste(_c6, (1322, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1322, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 54, 62)
    canvas.paste(_c7, (247, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 62)
    canvas.paste(_c8, (1260, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1260, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 48, 64)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/10_text_9.15.png
try:
    _c10 = get_crop(10, 94, 43)
    canvas.paste(_c10, (20, 17), _c10)
except Exception:
    pass
layout["9.15"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_03_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-5/17_text_Los_Angeles.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
