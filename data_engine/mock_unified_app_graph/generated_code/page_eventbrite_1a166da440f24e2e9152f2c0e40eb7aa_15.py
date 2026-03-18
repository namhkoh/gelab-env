# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_15
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17.png
# step_index: 15/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for Eventbrite-like page

w, h = canvas.size

# Colors
status_bar_color = "#d7d7d7"      # light gray status bar
hero_bg_color = "#e8f3ea"         # muted greenish for hero image background
page_bg = "#ffffff"               # white page background
divider_color = "#ececec"         # thin separators
card_pink = "#fff4f3"             # pale pink agenda card
card_pink_border = "#f6ded9"
accent_line = "#f5a89b"           # coral vertical accent in agenda card
card_top_strip = "#fff0ee"
shadow_color = "#efe8e8"

# 1) Overall page background (canvas already white, but ensure)
draw.rectangle((0, 0, w, h), fill=page_bg)

# 2) Status bar at top (~56px)
status_h = 56
draw.rectangle((0, 0, w, status_h), fill=status_bar_color)

# 3) Hero image banner background (placeholder for the image area)
hero_top = status_h
hero_bottom = 560
draw.rectangle((0, hero_top, w, hero_bottom), fill=hero_bg_color)

# Slight horizontal gradient effect on hero (subtle darker edges)
edge_width = 120
draw.rectangle((0, hero_top, edge_width, hero_bottom), fill="#e1ecdf")
draw.rectangle((w-edge_width, hero_top, w, hero_bottom), fill="#e1ecdf")

# 4) Divider under hero
draw.line((48, hero_bottom, w-48, hero_bottom), fill=divider_color, width=2)

# 5) Thin separator under the "refund policy" area (~y ≈ 1060)
sep1_y = 1060
draw.line((48, sep1_y, w-48, sep1_y), fill=divider_color, width=2)

# 6) Subtle section divider after "About this event" area (~y ≈ 2080)
sep2_y = 2080
draw.line((48, sep2_y, w-48, sep2_y), fill=divider_color, width=2)

# 7) Agenda area background card (rounded rectangle)
agenda_card_x1 = 48
agenda_card_x2 = w - 48
agenda_card_y1 = 2460
agenda_card_y2 = min(h - 40, agenda_card_y1 + 420)
draw.rounded_rectangle(
    (agenda_card_x1, agenda_card_y1, agenda_card_x2, agenda_card_y2),
    radius=28,
    fill=card_pink,
    outline=card_pink_border,
    width=1
)

# 7a) Soft shadow under agenda card (a faint rectangle)
shadow_y1 = agenda_card_y2 + 8
shadow_y2 = shadow_y1 + 10
draw.rectangle((agenda_card_x1 + 6, shadow_y1, agenda_card_x2 - 6, shadow_y2), fill=shadow_color)

# 7b) Coral vertical accent line inside the agenda card (left)
accent_x = agenda_card_x1 + 36
accent_margin_top = agenda_card_y1 + 28
accent_margin_bottom = agenda_card_y2 - 28
draw.rectangle((accent_x, accent_margin_top, accent_x + 8, accent_margin_bottom), fill=accent_line)

# 7c) Pale top strip inside the agenda card to separate header area
strip_x1 = agenda_card_x1 + 120
strip_x2 = agenda_card_x2 - 64
strip_y1 = agenda_card_y1 + 28
strip_y2 = strip_y1 + 42
draw.rectangle((strip_x1, strip_y1, strip_x2, strip_y2), fill=card_top_strip)

# 8) Light rounded container behind the small category/tag area (subtle background)
# This provides a subtle panel area under "About this event" without duplicating the detected tag itself.
about_panel_x1 = 40
about_panel_x2 = w - 40
about_panel_y1 = 1540
about_panel_y2 = 2020
draw.rounded_rectangle(
    (about_panel_x1, about_panel_y1, about_panel_x2, about_panel_y2),
    radius=12,
    fill=page_bg,
    outline=None
)

# 9) Subtle horizontal separators for content structure
draw.line((48, 1600, w-48, 1600), fill=divider_color, width=1)
draw.line((48, 1960, w-48, 1960), fill=divider_color, width=1)

# 10) Footer top separator (above bottom area)
footer_sep_y = h - 200
draw.line((48, footer_sep_y, w-48, footer_sep_y), fill=divider_color, width=1)

# NOTE: Do not draw any icons, labels, or interactive elements that will be pasted later.
# This file intentionally only draws background fills, cards, dividers, and large structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/00_icon_Sports_Fitness.png
try:
    _c0 = get_crop(0, 234, 144)
    canvas.paste(_c0, (48, 1965), _c0)
except Exception:
    pass
layout["Sports_&_Fitness"] = [48, 1965, 282, 2109]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/02_icon_Agenda.png
try:
    _c2 = get_crop(2, 255, 105)
    canvas.paste(_c2, (85, 2393), _c2)
except Exception:
    pass
layout["Agenda"] = [85, 2393, 340, 2498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/03_icon_Untitled_agenda.png
try:
    _c3 = get_crop(3, 375, 105)
    canvas.paste(_c3, (352, 2392), _c3)
except Exception:
    pass
layout["Untitled_agenda"] = [352, 2392, 727, 2497]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/05_icon_Ted_Young.png
try:
    _c5 = get_crop(5, 263, 131)
    canvas.paste(_c5, (213, 2775), _c5)
except Exception:
    pass
layout["Ted_Young"] = [213, 2775, 476, 2906]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/06_icon_5.31.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["5.31"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/07_icon_5.31.png
try:
    _c7 = get_crop(7, 65, 68)
    canvas.paste(_c7, (178, 1), _c7)
except Exception:
    pass
layout["5.31"] = [178, 1, 243, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/08_icon_5.31.png
try:
    _c8 = get_crop(8, 65, 69)
    canvas.paste(_c8, (112, 0), _c8)
except Exception:
    pass
layout["5.31"] = [112, 0, 177, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 55, 62)
    canvas.paste(_c9, (1318, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1318, 1, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 56, 64)
    canvas.paste(_c10, (246, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [246, 2, 302, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 68, 66)
    canvas.paste(_c11, (307, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [307, 1, 375, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 87, 59)
    canvas.paste(_c12, (1216, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1216, 3, 1303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 43, 53)
    canvas.paste(_c13, (1271, 7), _c13)
except Exception:
    pass
layout["icon_13"] = [1271, 7, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/14_icon_1_hrs_30_mins.png
try:
    _c14 = get_crop(14, 376, 73)
    canvas.paste(_c14, (55, 1201), _c14)
except Exception:
    pass
layout["1_hrs_30_mins"] = [55, 1201, 431, 1274]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/15_icon_Basics_of_Roller_Skating_balance_power.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1116, 108), _c15)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/16_text_5.31.png
try:
    _c16 = get_crop(16, 85, 43)
    canvas.paste(_c16, (22, 17), _c16)
except Exception:
    pass
layout["5.31"] = [22, 17, 107, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/17_text_Online_event.png
try:
    _c17 = get_crop(17, 274, 54)
    canvas.paste(_c17, (139, 1101), _c17)
except Exception:
    pass
layout["Online_event"] = [139, 1101, 413, 1155]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/18_text_Refund_policy.png
try:
    _c18 = get_crop(18, 299, 63)
    canvas.paste(_c18, (138, 1317), _c18)
except Exception:
    pass
layout["Refund_policy"] = [138, 1317, 437, 1380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/19_text_The_organizer_will_review_refund_request.png
try:
    _c19 = get_crop(19, 1344, 144)
    canvas.paste(_c19, (48, 1055), _c19)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1055, 1392, 1199]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/20_text_About_this_event.png
try:
    _c20 = get_crop(20, 454, 61)
    canvas.paste(_c20, (45, 1618), _c20)
except Exception:
    pass
layout["About_this_event"] = [45, 1618, 499, 1679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/21_text_Learn_how_to_improve_your_balance_and_po.png
try:
    _c21 = get_crop(21, 234, 144)
    canvas.paste(_c21, (48, 1965), _c21)
except Exception:
    pass
layout["Learn_how_to_improve_your"] = [48, 1965, 282, 2109]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/22_text_Read_more.png
try:
    _c22 = get_crop(22, 234, 144)
    canvas.paste(_c22, (48, 1965), _c22)
except Exception:
    pass
layout["Read_more"] = [48, 1965, 282, 2109]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/23_text_Agenda.png
try:
    _c23 = get_crop(23, 229, 75)
    canvas.paste(_c23, (42, 2227), _c23)
except Exception:
    pass
layout["Agenda"] = [42, 2227, 271, 2302]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/24_text_4.00_PM.png
try:
    _c24 = get_crop(24, 165, 43)
    canvas.paste(_c24, (221, 2596), _c24)
except Exception:
    pass
layout["4.00_PM"] = [221, 2596, 386, 2639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/25_text_5.30_PM.png
try:
    _c25 = get_crop(25, 170, 48)
    canvas.paste(_c25, (409, 2592), _c25)
except Exception:
    pass
layout["5.30_PM"] = [409, 2592, 579, 2640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/26_text_Your_instructor.png
try:
    _c26 = get_crop(26, 384, 64)
    canvas.paste(_c26, (217, 2664), _c26)
except Exception:
    pass
layout["Your_instructor"] = [217, 2664, 601, 2728]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_15_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-17/27_clickable_clickable_05.png
try:
    _c27 = get_crop(27, 1440, 594)
    canvas.paste(_c27, (0, 2366), _c27)
except Exception:
    pass
layout["clickable_05"] = [0, 2366, 1440, 2960]
