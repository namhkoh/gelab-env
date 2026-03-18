# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_16
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18.png
# step_index: 16/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile event page

# Colors
bg_offwhite = (250, 250, 253)       # overall page background (slightly off-white)
status_bar_gray = (190, 190, 190)   # status bar top
header_white = (255, 255, 255)      # header background
divider_gray = (235, 233, 240)      # subtle divider lines
image_dark = (40, 40, 40)           # dark gradient under hero image
image_mid = (90, 90, 90)
card_fill = (246, 244, 248)         # light card background (organizer card)
card_border = (232, 228, 238)       # card border
ticket_border = (59, 84, 255)       # blue outline for ticket/card
ticket_fill = (255, 255, 255)
shadow_color = (230, 229, 235)

W, H = canvas.size

# 1) Page background
draw.rectangle([0, 0, W, H], fill=bg_offwhite)

# 2) Status bar area at top (~56px)
status_h = 56
draw.rectangle([0, 0, W, status_h], fill=status_bar_gray)

# 3) Header / toolbar background (under status bar)
toolbar_h = 136  # includes some space for icons (we won't draw icons themselves)
draw.rectangle([0, status_h, W, status_h + toolbar_h], fill=header_white)
# subtle bottom divider for header
draw.line([48, status_h + toolbar_h, W - 48, status_h + toolbar_h], fill=divider_gray, width=1)

# 4) Hero image banner area (centered content area)
# draw a neutral placeholder background band for the hero image area (so pasted image will sit on it)
banner_top = status_h + 24
banner_left = 240
banner_right = W - 240
banner_bottom = banner_top + 420
draw.rectangle([banner_left, banner_top, banner_right, banner_bottom], fill=(245, 243, 246))

# 4a) Dark gradient overlay band beneath the hero image (to mimic the fade seen in the screenshot)
grad_top = banner_bottom - 60
grad_bottom = banner_bottom + 56
steps = 18
for i in range(steps):
    t = i / float(steps - 1)
    # interpolate between nearly transparent (lighter) and darker gray
    r = int(image_mid[0] * t + 245 * (1 - t))
    g = int(image_mid[1] * t + 243 * (1 - t))
    b = int(image_mid[2] * t + 246 * (1 - t))
    band_y0 = int(grad_top + (grad_bottom - grad_top) * (i / steps))
    band_y1 = int(grad_top + (grad_bottom - grad_top) * ((i + 1) / steps))
    draw.rectangle([banner_left, band_y0, banner_right, band_y1], fill=(r, g, b))

# 4b) small progress bars under banner (structural only — thin neutral bars)
bars_y = banner_bottom + 12
bar_w = 220
space = 20
start_x = (W - (bar_w * 4 + space * 3)) // 2
for i in range(4):
    bx = start_x + i * (bar_w + space)
    # lighter then darker for variety
    color = (230, 230, 230) if i != 1 else (210, 210, 210)
    draw.rectangle([bx, bars_y, bx + bar_w, bars_y + 10], fill=color, outline=None)

# subtle divider below banner area
draw.line([48, bars_y + 30, W - 48, bars_y + 30], fill=divider_gray, width=1)

# 5) Organizer card (rounded rectangle)
# Positioned so the organizer profile (detected) will be pasted on top of it
card_x0 = 48
card_x1 = W - 48
card_y0 = 1008
card_height = 160
card_y1 = card_y0 + card_height
card_radius = 24

# drop shadow (subtle)
draw.rectangle([card_x0 + 4, card_y0 + 6, card_x1 + 4, card_y1 + 6], fill=shadow_color)
# main card
draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=card_radius, fill=card_fill, outline=card_border, width=1)

# 6) Thin separators between content sections (rule lines)
sep1_y = card_y1 + 58
draw.line([48, sep1_y, W - 48, sep1_y], fill=divider_gray, width=1)

# 7) Refund/Info area (leave text area blank, but provide a subtle divider and spacing)
info_top = sep1_y + 24
draw.line([48, info_top + 120, W - 48, info_top + 120], fill=divider_gray, width=1)

# 8) "About this event" separator area (large section header space — structural only)
about_top = info_top + 160
draw.line([48, about_top, W - 48, about_top], fill=divider_gray, width=1)

# 9) Ticket selection card (rounded rectangle with colored outline)
ticket_x0 = 48
ticket_x1 = W - 48
ticket_y0 = 2380
ticket_height = 180
ticket_y1 = ticket_y0 + ticket_height
ticket_radius = 20

# subtle shadow under ticket card
draw.rectangle([ticket_x0 + 4, ticket_y0 + 6, ticket_x1 + 4, ticket_y1 + 6], fill=shadow_color)
# white interior
draw.rounded_rectangle([ticket_x0, ticket_y0, ticket_x1, ticket_y1], radius=ticket_radius, fill=ticket_fill, outline=ticket_border, width=6)

# inner horizontal divider inside ticket card (to separate title and price area)
inner_div_y = ticket_y0 + 86
draw.line([ticket_x0 + 28, inner_div_y, ticket_x1 - 28, inner_div_y], fill=(244, 244, 248), width=1)

# 10) Additional subtle separators above reserve area (but do NOT draw the reserve button itself)
reserve_gap_y = ticket_y1 + 28
draw.line([48, reserve_gap_y, W - 48, reserve_gap_y], fill=(245, 244, 246), width=1)

# End of structural drawing.
# (No text or icons are drawn; icon and text elements will be pasted on top later.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1068), _c0)
except Exception:
    pass
layout["Following"] = [946, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/04_icon_Health_Wellness.png
try:
    _c4 = get_crop(4, 234, 119)
    canvas.paste(_c4, (48, 2205), _c4)
except Exception:
    pass
layout["Health_&_Wellness"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 91, 102)
    canvas.paste(_c5, (1109, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1109, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 44, 69)
    canvas.paste(_c6, (1156, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1156, 1, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/07_icon_4.45.png
try:
    _c7 = get_crop(7, 65, 68)
    canvas.paste(_c7, (178, 0), _c7)
except Exception:
    pass
layout["4.45"] = [178, 0, 243, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/08_icon_More.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1116, 108), _c8)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/09_icon_4.45.png
try:
    _c9 = get_crop(9, 66, 70)
    canvas.paste(_c9, (111, 0), _c9)
except Exception:
    pass
layout["4.45"] = [111, 0, 177, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/10_icon_Share.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1260, 108), _c10)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 46, 65)
    canvas.paste(_c11, (1326, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1326, 3, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 56, 66)
    canvas.paste(_c12, (246, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [246, 1, 302, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 67, 66)
    canvas.paste(_c13, (308, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [308, 1, 375, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/14_icon_4.45.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (36, 108), _c14)
except Exception:
    pass
layout["4.45"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 98, 64)
    canvas.paste(_c15, (1216, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1216, 2, 1314, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 53, 67)
    canvas.paste(_c16, (382, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 1, 435, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/17_icon_Personal_health.png
try:
    _c17 = get_crop(17, 234, 119)
    canvas.paste(_c17, (48, 2205), _c17)
except Exception:
    pass
layout["Personal_health"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/18_icon_The_organizer_will_review_refund_request.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1295), _c18)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/19_text_Friday_May_10_._H_00_AM.png
try:
    _c19 = get_crop(19, 314, 144)
    canvas.paste(_c19, (288, 1068), _c19)
except Exception:
    pass
layout["Friday;_May_10_._H:00_AM"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/20_text_Wellness_in_Action_Core_Work.png
try:
    _c20 = get_crop(20, 314, 144)
    canvas.paste(_c20, (288, 1068), _c20)
except Exception:
    pass
layout["Wellness_in_Action:_Core_"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/21_text_Danielle_Smith.png
try:
    _c21 = get_crop(21, 314, 144)
    canvas.paste(_c21, (288, 1068), _c21)
except Exception:
    pass
layout["Danielle_Smith"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/22_text_General_Admission.png
try:
    _c22 = get_crop(22, 75, 72)
    canvas.paste(_c22, (249, 2588), _c22)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/23_text_Free.png
try:
    _c23 = get_crop(23, 105, 48)
    canvas.paste(_c23, (116, 2599), _c23)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_16_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-18/24_clickable_Organizer_profile_picture.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (96, 1067), _c24)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]
