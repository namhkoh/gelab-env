# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_09
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12.png
# step_index: 9/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structure for the mobile UI page
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_offwhite = (249, 249, 249)
status_bg = (238, 238, 238)
hero_deep = (12, 28, 74)        # deep navy for hero banner
hero_accent = (12, 94, 196)     # bright blue accent near top
card_white = (255, 255, 255)
divider_gray = (224, 224, 224)
shadow_gray = (230, 230, 230)

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_offwhite)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bg)

# Hero image background band (below status bar)
hero_top = status_h
hero_bottom = 420
draw.rectangle([(0, hero_top), (W, hero_bottom)], fill=hero_deep)

# Subtle blue accent stripe near top edge of hero (thin)
accent_h0 = hero_top + 10
accent_h1 = accent_h0 + 12
draw.rectangle([(0, accent_h0), (W, accent_h1)], fill=hero_accent)

# Add a slanted white overlay at the bottom of the hero to create diagonal cut
# This forms the transition into the white content card below.
slant = [
    (0, hero_bottom - 30),
    (W, hero_bottom - 100),
    (W, hero_bottom + 180),
    (0, hero_bottom + 80)
]
draw.polygon(slant, fill=card_white)

# Main top content card (white rounded rectangle) that sits partly over the hero.
# Stop this card at y = 1146 so we don't draw over the area where the two action buttons are auto-pasted.
card_left, card_right = 24, W - 24
card_top = 360
card_bottom = 1146  # do not draw into the action-buttons area (buttons occupy y=1146..1299)
card_radius = 20

# Subtle shadow under the card (thin band)
shadow_rect = [(card_left + 2, card_top + 8), (card_right - 2, card_bottom + 12)]
draw.rectangle(shadow_rect, fill=shadow_gray)

# White rounded card
try:
    draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                           radius=card_radius, fill=card_white)
except Exception:
    # Fallback if rounded_rectangle not supported
    draw.rectangle([(card_left, card_top), (card_right, card_bottom)], fill=card_white)

# Divider line under the card area (subtle)
div_y = card_bottom + 2
draw.line([(24, div_y), (W - 24, div_y)], fill=divider_gray, width=1)

# Area below the buttons: large content area (white background) starting just below the buttons
# Buttons occupy y from 1146 to 1299 (height 153). Start background at 1299 to avoid overlapping them.
content_top = 1299
draw.rectangle([(0, content_top), (W, H)], fill=card_white)

# Subtle horizontal separators for sections in the content area
separators = [
    content_top + 80,   # after "Location" block
    content_top + 200,  # after "Get directions" / "More events..." area
    content_top + 360,  # before "Performers" heading
    content_top + 760,  # between performer list rows
    content_top + 1060, # lower grouping
]
for y in separators:
    draw.line([(24, y), (W - 24, y)], fill=divider_gray, width=1)

# "Performers" card header area: leave white space but add a faint off-white band to separate sections
performers_header_top = content_top + 320
performers_header_bottom = performers_header_top + 88
draw.rectangle([(0, performers_header_top), (W, performers_header_bottom)], fill=bg_offwhite)

# Add a subtle left inset rule to visually group the performers list (very light)
inset_x = 24
draw.line([(inset_x, performers_header_bottom), (W - inset_x, performers_header_bottom)], fill=divider_gray, width=1)

# Bottom area faint divider near end of page
draw.line([(24, H - 260), (W - 24, H - 260)], fill=divider_gray, width=1)

# Small decorative horizontal accent near top of the hero to mimic the blue stadium LED strip
led_y = hero_top + 46
for i in range(6):
    x0 = 40 + i * 220
    x1 = x0 + 160
    draw.rectangle([(x0, led_y), (x1, led_y + 8)], fill=(32, 150, 255))

# Ensure edges remain crisp by drawing a faint border on the very top of content area
draw.line([(0, content_top), (W, content_top)], fill=divider_gray, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/00_icon_Untrack_event.png
try:
    _c0 = get_crop(0, 498, 153)
    canvas.paste(_c0, (60, 1146), _c0)
except Exception:
    pass
layout["Untrack_event"] = [60, 1146, 558, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/01_icon_Share.png
try:
    _c1 = get_crop(1, 312, 153)
    canvas.paste(_c1, (606, 1146), _c1)
except Exception:
    pass
layout["Share"] = [606, 1146, 918, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/02_icon_Eastern_Conference_First_Round.png
try:
    _c2 = get_crop(2, 312, 153)
    canvas.paste(_c2, (606, 1146), _c2)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [606, 1146, 918, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/03_icon_6.58_my.png
try:
    _c3 = get_crop(3, 61, 68)
    canvas.paste(_c3, (112, 1), _c3)
except Exception:
    pass
layout["6.58_my"] = [112, 1, 173, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/04_icon_6.58_my.png
try:
    _c4 = get_crop(4, 51, 66)
    canvas.paste(_c4, (183, 3), _c4)
except Exception:
    pass
layout["6.58_my"] = [183, 3, 234, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 67)
    canvas.paste(_c5, (242, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 3, 305, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/06_icon_24_events.png
try:
    _c6 = get_crop(6, 1416, 179)
    canvas.paste(_c6, (12, 2697), _c6)
except Exception:
    pass
layout["24_events"] = [12, 2697, 1428, 2876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 69)
    canvas.paste(_c7, (1154, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1154, 2, 1202, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/08_icon_Philadelphia_76ers.png
try:
    _c8 = get_crop(8, 1416, 179)
    canvas.paste(_c8, (12, 2160), _c8)
except Exception:
    pass
layout["Philadelphia_76ers"] = [12, 2160, 1428, 2339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 69)
    canvas.paste(_c9, (1319, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1319, 1, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/10_icon_215_events.png
try:
    _c10 = get_crop(10, 1416, 179)
    canvas.paste(_c10, (12, 2518), _c10)
except Exception:
    pass
layout["215_events"] = [12, 2518, 1428, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/11_icon_NBA_Eastern_Conference_First_Round.png
try:
    _c11 = get_crop(11, 1416, 179)
    canvas.paste(_c11, (12, 2518), _c11)
except Exception:
    pass
layout["NBA_Eastern_Conference_Fi"] = [12, 2518, 1428, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/12_icon_Performers.png
try:
    _c12 = get_crop(12, 1416, 179)
    canvas.paste(_c12, (12, 2160), _c12)
except Exception:
    pass
layout["Performers"] = [12, 2160, 1428, 2339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 57, 68)
    canvas.paste(_c13, (314, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [314, 2, 371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/14_icon_18_events.png
try:
    _c14 = get_crop(14, 1416, 179)
    canvas.paste(_c14, (12, 2339), _c14)
except Exception:
    pass
layout["18_events"] = [12, 2339, 1428, 2518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 60, 70)
    canvas.paste(_c15, (1211, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1211, 1, 1271, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/16_icon_New_York_Knicks.png
try:
    _c16 = get_crop(16, 1416, 179)
    canvas.paste(_c16, (12, 2339), _c16)
except Exception:
    pass
layout["New_York_Knicks"] = [12, 2339, 1428, 2518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/17_icon_6.58_my.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (24, 84), _c17)
except Exception:
    pass
layout["6.58_my"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 47, 66)
    canvas.paste(_c18, (1270, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1270, 3, 1317, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/19_icon_New_York_Knicks_at_Philadelphia.png
try:
    _c19 = get_crop(19, 498, 153)
    canvas.paste(_c19, (60, 1146), _c19)
except Exception:
    pass
layout["New_York_Knicks_at_Philad"] = [60, 1146, 558, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 67)
    canvas.paste(_c20, (382, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/21_icon_NBA_Eastern_Conference_First_Round.png
try:
    _c21 = get_crop(21, 1416, 179)
    canvas.paste(_c21, (12, 2697), _c21)
except Exception:
    pass
layout["NBA_Eastern_Conference_Fi"] = [12, 2697, 1428, 2876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/22_icon_New_York_Knicks.png
try:
    _c22 = get_crop(22, 359, 56)
    canvas.paste(_c22, (244, 2372), _c22)
except Exception:
    pass
layout["New_York_Knicks"] = [244, 2372, 603, 2428]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/23_icon_NBA_Playoffs.png
try:
    _c23 = get_crop(23, 284, 55)
    canvas.paste(_c23, (244, 2552), _c23)
except Exception:
    pass
layout["NBA_Playoffs"] = [244, 2552, 528, 2607]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/24_text_Location.png
try:
    _c24 = get_crop(24, 212, 52)
    canvas.paste(_c24, (53, 1432), _c24)
except Exception:
    pass
layout["Location"] = [53, 1432, 265, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/25_text_Wells_Fargo_Center.png
try:
    _c25 = get_crop(25, 410, 63)
    canvas.paste(_c25, (55, 1546), _c25)
except Exception:
    pass
layout["Wells_Fargo_Center"] = [55, 1546, 465, 1609]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/26_text_Philadelphia_PA_19148.png
try:
    _c26 = get_crop(26, 445, 57)
    canvas.paste(_c26, (53, 1621), _c26)
except Exception:
    pass
layout["Philadelphia,_PA_19148"] = [53, 1621, 498, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/27_text_Get_directions.png
try:
    _c27 = get_crop(27, 1440, 113)
    canvas.paste(_c27, (0, 1721), _c27)
except Exception:
    pass
layout["Get_directions"] = [0, 1721, 1440, 1834]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/28_text_More_events_at_Wells_Fargo_Center.png
try:
    _c28 = get_crop(28, 1440, 113)
    canvas.paste(_c28, (0, 1834), _c28)
except Exception:
    pass
layout["More_events_at_Wells_Farg"] = [0, 1834, 1440, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_09_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-12/29_text_Performers.png
try:
    _c29 = get_crop(29, 255, 52)
    canvas.paste(_c29, (56, 2061), _c29)
except Exception:
    pass
layout["Performers"] = [56, 2061, 311, 2113]
