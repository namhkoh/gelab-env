# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_14
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16.png
# step_index: 14/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960)
w, h = canvas.size

# Colors
status_bg = "#D0D0D0"        # status bar light gray
divider = "#E6E6E9"          # light divider
page_bg = "#FFFFFF"          # main background (keeps canvas white)
hero_top = (255, 255, 255)   # hero gradient top (white)
hero_bottom = (245, 245, 250) # hero gradient bottom (very light cool)
hero_fade = "#EDEFF2"        # bottom fade under image
card_shadow = "#E8E6EA"
card_fill = "#F7F5FA"        # pale lilac card fill
card_border = "#ECE8F2"
ticket_border = "#2D41FF"    # blue border for ticket selection
ticket_fill = "#FFFFFF"
muted_line = "#F0F0F3"

# 1) Status bar area (approx ~50px tall)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bg)
# subtle bottom divider
draw.line([(0, status_h), (w, status_h)], fill=divider, width=1)

# 2) Hero/banner area with vertical subtle gradient (under the top toolbar)
hero_y0 = status_h + 8
hero_h = 460
hero_y1 = hero_y0 + hero_h
# vertical gradient
r0, g0, b0 = hero_top
r1, g1, b1 = hero_bottom
for i in range(hero_h):
    t = i / max(1, hero_h - 1)
    r = int(r0 + (r1 - r0) * t)
    g = int(g0 + (g1 - g0) * t)
    b = int(b0 + (b1 - b0) * t)
    draw.line([(0, hero_y0 + i), (w, hero_y0 + i)], fill=(r, g, b))

# subtle bottom fade bar under the image area (dark-to-light illusion)
fade_h = 40
fade_y0 = hero_y1 - fade_h
# simple solid fade band
draw.rectangle([(0, fade_y0), (w, hero_y1)], fill=hero_fade)

# 3) Small progress indicator bars centered beneath hero image (series of small rounded rects)
bars_y = hero_y1 - 18
bar_count = 7
total_width = 560
gap = 18
bar_widths = [80, 60, 40, 140, 60, 40, 120]  # varying widths like screenshot
start_x = (w - total_width) // 2
x = start_x
for bw in bar_widths:
    rx0 = x
    rx1 = x + bw
    # active/inactive effect: the center bar darker
    color = "#DCDCDF" if abs((rx0 + rx1) / 2 - w/2) > 60 else "#BDBCC0"
    draw.rounded_rectangle([(rx0, bars_y), (rx1, bars_y + 8)], radius=4, fill=color)
    x += bw + gap

# thin separator under hero area
sep_y = hero_y1 + 8
draw.line([(48, sep_y), (w - 48, sep_y)], fill=muted_line, width=1)

# 4) Organizer/profile card (rounded rectangle) below title area
card_x = 48
card_w = w - 2 * card_x
card_y = sep_y + 64
card_h = 140
# shadow
shadow_offset = 6
draw.rounded_rectangle(
    [(card_x + shadow_offset, card_y + shadow_offset), (card_x + card_w + shadow_offset, card_y + card_h + shadow_offset)],
    radius=20, fill=card_shadow
)
# card body
draw.rounded_rectangle([(card_x, card_y), (card_x + card_w, card_y + card_h)], radius=20, fill=card_fill, outline=card_border, width=2)

# inside subtle divider (to suggest separation of avatar + follow button area)
inner_div_y = card_y + 16
draw.line([(card_x + 16, card_y + card_h - 16), (card_x + card_w - 16, card_y + card_h - 16)], fill="#F1F0F4", width=1)

# 5) Info rows separators (thin horizontal separators under the organizer card)
info_start_y = card_y + card_h + 32
draw.line([(48, info_start_y), (w - 48, info_start_y)], fill=muted_line, width=1)

# a couple of subtle icon+text row separators (visual structure only)
row_y = info_start_y + 96
draw.line([(48, row_y), (w - 48, row_y)], fill=muted_line, width=1)

row_y2 = row_y + 96
draw.line([(48, row_y2), (w - 48, row_y2)], fill=muted_line, width=1)

# 6) "About this event" section divider
about_y = row_y2 + 96
draw.line([(48, about_y), (w - 48, about_y)], fill=muted_line, width=1)

# 7) Tickets selection card (rounded rectangle with blue border) above Reserve area
ticket_y = about_y + 220
ticket_h = 220
ticket_x = 48
ticket_w = w - 2 * ticket_x
# card shadow
draw.rounded_rectangle([(ticket_x + 4, ticket_y + 6), (ticket_x + ticket_w + 4, ticket_y + ticket_h + 6)], radius=18, fill="#F3F3F5")
# card with border
draw.rounded_rectangle([(ticket_x, ticket_y), (ticket_x + ticket_w, ticket_y + ticket_h)], radius=18, fill=ticket_fill, outline=ticket_border, width=6)

# inner lighter divider in ticket card
draw.line([(ticket_x + 28, ticket_y + 80), (ticket_x + ticket_w - 28, ticket_y + 80)], fill="#F0F4FF", width=2)

# small rounded pill to represent price pill background (but not the detected text)
pill_w = 220
pill_h = 48
pill_x = ticket_x + 24
pill_y = ticket_y + 120
draw.rounded_rectangle([(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)], radius=24, fill="#FAFBFF", outline="#E8ECFF", width=1)

# 8) Large bottom safe divider just above the Reserve button area (do not draw the reserve button)
bottom_div_y = ticket_y + ticket_h + 24
draw.line([(24, bottom_div_y), (w - 24, bottom_div_y)], fill=divider, width=1)

# 9) Final subtle shadow band above bottom area (gives depth before Reserve area which will be pasted)
shadow_band_h = 12
draw.rectangle([(0, bottom_div_y), (w, bottom_div_y + shadow_band_h)], fill="#F6F5F7")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 114, 107)
    canvas.paste(_c1, (987, 2439), _c1)
except Exception:
    pass
layout["icon_1"] = [987, 2439, 1101, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1440, 636)
    canvas.paste(_c2, (0, 2324), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 109, 105)
    canvas.paste(_c3, (1214, 2441), _c3)
except Exception:
    pass
layout["icon_3"] = [1214, 2441, 1323, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/04_icon_Health_Wellness.png
try:
    _c4 = get_crop(4, 234, 119)
    canvas.paste(_c4, (48, 2205), _c4)
except Exception:
    pass
layout["Health_&_Wellness"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 91, 102)
    canvas.paste(_c5, (1109, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1109, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 47, 69)
    canvas.paste(_c6, (1155, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1155, 1, 1202, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/07_icon_4.45.png
try:
    _c7 = get_crop(7, 65, 68)
    canvas.paste(_c7, (178, 0), _c7)
except Exception:
    pass
layout["4.45"] = [178, 0, 243, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/08_icon_4.45.png
try:
    _c8 = get_crop(8, 66, 70)
    canvas.paste(_c8, (111, 0), _c8)
except Exception:
    pass
layout["4.45"] = [111, 0, 177, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/09_icon_Share.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1260, 108), _c9)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/10_icon_More.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1116, 108), _c10)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 46, 64)
    canvas.paste(_c11, (1326, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1326, 3, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 55, 68)
    canvas.paste(_c12, (246, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [246, 0, 301, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 66, 65)
    canvas.paste(_c13, (308, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [308, 1, 374, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/14_icon_4.45.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (36, 108), _c14)
except Exception:
    pass
layout["4.45"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 98, 65)
    canvas.paste(_c15, (1215, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1215, 1, 1313, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 67)
    canvas.paste(_c16, (382, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 1, 434, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/17_icon_Danielle_Smith.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (96, 1067), _c17)
except Exception:
    pass
layout["Danielle_Smith"] = [96, 1067, 240, 1211]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/18_icon_Personal_health.png
try:
    _c18 = get_crop(18, 234, 119)
    canvas.paste(_c18, (48, 2205), _c18)
except Exception:
    pass
layout["Personal_health"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/19_text_Friday_May_10_._H_00_AM.png
try:
    _c19 = get_crop(19, 314, 144)
    canvas.paste(_c19, (288, 1068), _c19)
except Exception:
    pass
layout["Friday;_May_10_._H:00_AM"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/20_text_Wellness_in_Action_Core_Work.png
try:
    _c20 = get_crop(20, 314, 144)
    canvas.paste(_c20, (288, 1068), _c20)
except Exception:
    pass
layout["Wellness_in_Action:_Core_"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/21_text_Danielle_Smith.png
try:
    _c21 = get_crop(21, 314, 144)
    canvas.paste(_c21, (288, 1068), _c21)
except Exception:
    pass
layout["Danielle_Smith"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/22_text_USO_Warrior_and_Family_Center_at_Fort_Be.png
try:
    _c22 = get_crop(22, 1344, 144)
    canvas.paste(_c22, (48, 1295), _c22)
except Exception:
    pass
layout["USO_Warrior_and_Family_Ce"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/23_text_45_mins.png
try:
    _c23 = get_crop(23, 179, 50)
    canvas.paste(_c23, (139, 1452), _c23)
except Exception:
    pass
layout["45_mins"] = [139, 1452, 318, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/24_text_Refund_policy.png
try:
    _c24 = get_crop(24, 299, 63)
    canvas.paste(_c24, (138, 1558), _c24)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/25_text_The_organizer_will_review_refund_request.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 1295), _c25)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/26_text_General_Admission.png
try:
    _c26 = get_crop(26, 234, 119)
    canvas.paste(_c26, (48, 2205), _c26)
except Exception:
    pass
layout["General_Admission"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_14_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-16/27_text_Free.png
try:
    _c27 = get_crop(27, 105, 48)
    canvas.paste(_c27, (116, 2599), _c27)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]
