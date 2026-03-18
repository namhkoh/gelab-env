# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_16
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18.png
# step_index: 16/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI mock
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full background (dominant white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top ~56px) - muted green-gray as in screenshot
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill="#9ea79f")

# Notification banner below status bar (~56..180) - pale mint/green
notif_top = status_h
notif_bottom = 180
draw.rectangle((0, notif_top, 1440, notif_bottom), fill="#eaf6ec")
# subtle divider lines for the notification banner
draw.line((24, notif_bottom, 1416, notif_bottom), fill="#dbe8df", width=1)
draw.line((24, notif_top, 1416, notif_top), fill="#dbe8df", width=1)

# Header image area (banner) - represent with a warm desaturated rectangle
banner_top = notif_bottom
banner_bottom = 460
draw.rectangle((0, banner_top, 1440, banner_bottom), fill="#bfa07c")
# subtle darker top and bottom edges for separation
draw.line((0, banner_bottom, 1440, banner_bottom), fill="#d9c3a4", width=1)
draw.line((0, banner_top, 1440, banner_top), fill="#caa883", width=1)

# Main content separators and structural lines
# thin divider under banner
draw.line((48, banner_bottom + 8, 1392, banner_bottom + 8), fill="#ececec", width=1)

# Separator under the "Refund policy" block area (~y = 1320)
sep1_y = 1320
draw.line((48, sep1_y, 1392, sep1_y), fill="#efefef", width=1)

# Large section divider before Agenda (~y = 2220)
sep2_y = 2220
draw.line((42, sep2_y, 1398, sep2_y), fill="#efefef", width=1)

# "Agenda" area background (keeps content readable) - leave text/icons out (they'll be pasted)
# Draw ticket card shadow
card_left = 42
card_top = 2340
card_right = 1398
card_bottom = 2620
shadow_offset = 8
draw.rounded_rectangle(
    (card_left + shadow_offset, card_top + shadow_offset, card_right + shadow_offset, card_bottom + shadow_offset),
    radius=24, fill="#f2f2f2"
)

# Ticket card background with blue outline (rounded)
card_outline_color = "#2b59d8"
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=24, fill="#FFFFFF", outline=card_outline_color, width=6
)

# Inner subtle divider inside card (to indicate content separation)
inner_div_y = card_top + 120
draw.line((card_left + 28, inner_div_y, card_right - 28, inner_div_y), fill="#f1f3fb", width=2)

# Light rounded badge background areas used across the page (decorative only, not overlapping detected chips)
# (Placeholders positioned to avoid duplicating detected chip at ~y=1965)
badge_x = 48
badge_y = 1880  # kept above the detected chip area to avoid overlap
draw.rounded_rectangle((badge_x, badge_y, badge_x + 220, badge_y + 56), radius=28, fill="#f3f5f8")

# Subtle horizontal rule above checkout area (keeps spacing, placed above the checkout button area)
hr_y = 2700
draw.line((48, hr_y, 1392, hr_y), fill="#efecec", width=1)

# Light bottom area fill (page bottom subtle tint to separate from button area)
bottom_strip_top = 2888
draw.rectangle((0, bottom_strip_top, 1440, 2960), fill="#ffffff")

# Final subtle vertical padding guides (thin faint lines at content margins for alignment)
draw.line((48, 0, 48, 2960), fill="#ffffff", width=1)   # left margin guide (invisible white, structural)
draw.line((1392, 0, 1392, 2960), fill="#ffffff", width=1)  # right margin guide

# (No text or icons drawn here; those will be pasted afterwards at detected positions.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/00_icon_Sports_Fitness.png
try:
    _c0 = get_crop(0, 234, 144)
    canvas.paste(_c0, (48, 1965), _c0)
except Exception:
    pass
layout["Sports_&_Fitness"] = [48, 1965, 282, 2109]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/02_icon_Check_out_for_S17.85.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Check_out_for_S17.85"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 91, 103)
    canvas.paste(_c4, (1108, 2442), _c4)
except Exception:
    pass
layout["icon_4"] = [1108, 2442, 1199, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 56, 56)
    canvas.paste(_c5, (312, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [312, 6, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/06_icon_Dismiss_notification.png
try:
    _c6 = get_crop(6, 142, 142)
    canvas.paste(_c6, (1251, 97), _c6)
except Exception:
    pass
layout["Dismiss_notification"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 52)
    canvas.paste(_c7, (252, 8), _c7)
except Exception:
    pass
layout["icon_7"] = [252, 8, 298, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/08_icon_5.32.png
try:
    _c8 = get_crop(8, 61, 61)
    canvas.paste(_c8, (180, 2), _c8)
except Exception:
    pass
layout["5.32"] = [180, 2, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/09_icon_5.32.png
try:
    _c9 = get_crop(9, 61, 63)
    canvas.paste(_c9, (113, 1), _c9)
except Exception:
    pass
layout["5.32"] = [113, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 45, 62)
    canvas.paste(_c10, (1325, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [1325, 3, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 63, 61)
    canvas.paste(_c11, (1211, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1211, 3, 1274, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 44, 63)
    canvas.paste(_c12, (1269, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1269, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 61)
    canvas.paste(_c13, (382, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [382, 3, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/14_icon_5.32.png
try:
    _c14 = get_crop(14, 91, 60)
    canvas.paste(_c14, (17, 4), _c14)
except Exception:
    pass
layout["5.32"] = [17, 4, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/15_icon_1_hrs_30_mins.png
try:
    _c15 = get_crop(15, 381, 74)
    canvas.paste(_c15, (47, 1201), _c15)
except Exception:
    pass
layout["1_hrs_30_mins"] = [47, 1201, 428, 1275]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/16_icon_Learn_how_to_improve_your_balance_and_po.png
try:
    _c16 = get_crop(16, 234, 144)
    canvas.paste(_c16, (48, 1965), _c16)
except Exception:
    pass
layout["Learn_how_to_improve_your"] = [48, 1965, 282, 2109]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/17_icon_S15.00.png
try:
    _c17 = get_crop(17, 75, 72)
    canvas.paste(_c17, (306, 2588), _c17)
except Exception:
    pass
layout["S15.00"] = [306, 2588, 381, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/18_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (36, 108), _c18)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/19_text_Monday_May_6_._4_00_PM.png
try:
    _c19 = get_crop(19, 621, 79)
    canvas.paste(_c19, (44, 757), _c19)
except Exception:
    pass
layout["Monday;_May_6_._4:00_PM"] = [44, 757, 665, 836]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/20_text_Basics_of_Roller_Skating_balance_power.png
try:
    _c20 = get_crop(20, 1344, 144)
    canvas.paste(_c20, (48, 1055), _c20)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [48, 1055, 1392, 1199]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/21_text_Online_event.png
try:
    _c21 = get_crop(21, 274, 54)
    canvas.paste(_c21, (139, 1101), _c21)
except Exception:
    pass
layout["Online_event"] = [139, 1101, 413, 1155]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/22_text_Refund_policy.png
try:
    _c22 = get_crop(22, 299, 63)
    canvas.paste(_c22, (138, 1317), _c22)
except Exception:
    pass
layout["Refund_policy"] = [138, 1317, 437, 1380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/23_text_The_organizer_will_review_refund_request.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1055), _c23)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1055, 1392, 1199]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/24_text_Agenda.png
try:
    _c24 = get_crop(24, 229, 75)
    canvas.paste(_c24, (42, 2227), _c24)
except Exception:
    pass
layout["Agenda"] = [42, 2227, 271, 2302]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/25_text_General_Admission.png
try:
    _c25 = get_crop(25, 75, 72)
    canvas.paste(_c25, (306, 2588), _c25)
except Exception:
    pass
layout["General_Admission"] = [306, 2588, 381, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_16_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-18/26_text_S15.00.png
try:
    _c26 = get_crop(26, 163, 57)
    canvas.paste(_c26, (113, 2592), _c26)
except Exception:
    pass
layout["S15.00"] = [113, 2592, 276, 2649]
