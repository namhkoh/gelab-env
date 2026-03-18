# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_08
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10.png
# step_index: 8/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the Event page UI
w, h = canvas.size

# Clear canvas (white)
draw.rectangle((0, 0, w, h), fill="#ffffff")

# Status bar area (top ~96px)
status_h = 96
draw.rectangle((0, 0, w, status_h), fill="#e6e6e6")
# subtle bottom divider for status bar
draw.line((24, status_h - 1, w - 24, status_h - 1), fill="#d0d0d0", width=1)

# Hero/header image background area (green blurred-style band)
hero_top = status_h
hero_bottom = 560
# base soft green
draw.rectangle((0, hero_top, w, hero_bottom), fill="#cfeed6")
# darker central band behind where the main image will be pasted
center_w = 480
center_left = (w // 2) - (center_w // 2)
center_right = center_left + center_w
draw.rectangle((center_left, hero_top, center_right, hero_bottom), fill="#2fa86f")
# subtle left and right vignette blocks to emulate blurred edges
draw.rectangle((0, hero_top, center_left, hero_bottom), fill="#a7db9e")
draw.rectangle((center_right, hero_top, w, hero_bottom), fill="#98d18e")

# Slight horizontal divider under hero
divider_y = hero_bottom + 8
draw.line((24, divider_y, w - 24, divider_y), fill="#f0f0f0", width=1)

# Main content background (white) already present; draw a very subtle large card background strip
content_top = hero_bottom + 24
draw.rectangle((0, content_top, w, h), fill="#ffffff")

# "Ticket sales" badge area background - do not draw text, only a tiny pill background for layout (left side)
# Position around where the detected badge is located (approx)
badge_x0 = 40
badge_y0 = 740
badge_x1 = badge_x0 + 360
badge_y1 = badge_y0 + 56
draw.rounded_rectangle((badge_x0, badge_y0, badge_x1, badge_y1), radius=28, fill="#f3e9ff")  # soft lavender pill

# Organizer card (rounded rectangle background behind organizer name + Follow button)
org_card_x0 = 40
org_card_x1 = w - 40
org_card_y0 = 1200
org_card_y1 = org_card_y0 + 160
draw.rounded_rectangle((org_card_x0, org_card_y0, org_card_x1, org_card_y1), radius=22, fill="#f7f7fb")
# subtle inner border
draw.rounded_rectangle((org_card_x0 + 2, org_card_y0 + 2, org_card_x1 - 2, org_card_y1 - 2), radius=20, outline="#e6e6ee", width=2)

# Small divider under the organizer/card area
draw.line((48, org_card_y1 + 28, w - 48, org_card_y1 + 28), fill="#efeff2", width=1)

# Info rows area (location / duration / refund) - draw light icon-column guide blocks (no icons/text)
info_start_y = org_card_y1 + 48
row_h = 72
# For three rows, draw subtle left-side round icon background placeholders (keeps structure)
for i in range(3):
    y0 = info_start_y + i * (row_h + 8)
    # icon placeholder circle area (only background, not an icon)
    cx = 56
    cy = y0 + row_h // 2
    r = 22
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#f6f6f9")

# Thin separator before "About this event"
about_div_y = info_start_y + 3 * (row_h + 8) + 24
draw.line((40, about_div_y, w - 40, about_div_y), fill="#efeff2", width=1)

# "About this event" header area (do not draw text) - draw a subtle bounding space
about_top = about_div_y + 24
draw.rectangle((40, about_top, w - 40, about_top + 120), fill="#ffffff")

# Category pill background (behind category labels) - left-aligned
cat_pill_x0 = 60
cat_pill_y0 = about_top + 40
cat_pill_x1 = cat_pill_x0 + 540
cat_pill_y1 = cat_pill_y0 + 56
draw.rounded_rectangle((cat_pill_x0, cat_pill_y0, cat_pill_x1, cat_pill_y1), radius=28, fill="#f3f4f6")

# Large horizontal separator line between content sections
draw.line((24, cat_pill_y1 + 40, w - 24, cat_pill_y1 + 40), fill="#efeff2", width=1)

# Ticket selection card (rounded with blue border) - no inner text/buttons
ticket_card_x0 = 40
ticket_card_x1 = w - 40
ticket_card_y0 = 2360
ticket_card_y1 = 2660
# white fill card
draw.rounded_rectangle((ticket_card_x0, ticket_card_y0, ticket_card_x1, ticket_card_y1), radius=20, fill="#ffffff", outline="#3b63ff", width=6)
# subtle inner divider area inside ticket card
inner_pad = 28
draw.line((ticket_card_x0 + inner_pad, ticket_card_y0 + 120, ticket_card_x1 - inner_pad, ticket_card_y0 + 120), fill="#f1f2f6", width=1)

# Small rounded background behind the quantity control area (right side of ticket card)
qty_bg_w = 120
qty_bg_h = 80
qty_bg_x1 = ticket_card_x1 - inner_pad
qty_bg_x0 = qty_bg_x1 - qty_bg_w
qty_bg_y0 = ticket_card_y0 + 40
qty_bg_y1 = qty_bg_y0 + qty_bg_h
draw.rounded_rectangle((qty_bg_x0, qty_bg_y0, qty_bg_x1, qty_bg_y1), radius=16, fill="#f7f7fb")

# Reserve button area is detected and will be pasted; draw only a faint drop shadow behind it (do not draw the button itself)
reserve_shadow_x0 = 72
reserve_shadow_x1 = w - 72
reserve_shadow_y0 = 2756 - 6
reserve_shadow_y1 = reserve_shadow_y0 + 140 + 6
draw.rectangle((reserve_shadow_x0, reserve_shadow_y0, reserve_shadow_x1, reserve_shadow_y1), fill="#f3e8df")

# Final subtle structural separators near bottom
draw.line((40, ticket_card_y1 + 40, w - 40, ticket_card_y1 + 40), fill="#efeff2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1290), _c0)
except Exception:
    pass
layout["Following"] = [946, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/03_icon_Decrease.png
try:
    _c3 = get_crop(3, 99, 96)
    canvas.paste(_c3, (996, 2444), _c3)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/04_icon_5.10_my.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["5.10_my"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 90, 101)
    canvas.paste(_c6, (1109, 2443), _c6)
except Exception:
    pass
layout["icon_6"] = [1109, 2443, 1199, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/07_icon_Reserve_a_spot.png
try:
    _c7 = get_crop(7, 1296, 132)
    canvas.paste(_c7, (72, 2756), _c7)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/08_icon_Ticket_sales_end_soon.png
try:
    _c8 = get_crop(8, 547, 84)
    canvas.paste(_c8, (40, 753), _c8)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/09_icon_The_organizer_will_review_refund_request.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 1517), _c9)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/10_icon_5.10_my.png
try:
    _c10 = get_crop(10, 61, 66)
    canvas.paste(_c10, (181, 1), _c10)
except Exception:
    pass
layout["5.10_my"] = [181, 1, 242, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/11_icon_IO_O0_AM.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1116, 108), _c11)
except Exception:
    pass
layout["IO:O0_AM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 65)
    canvas.paste(_c12, (248, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [248, 1, 301, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 97, 60)
    canvas.paste(_c13, (1217, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1217, 3, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/14_icon_Free.png
try:
    _c14 = get_crop(14, 134, 106)
    canvas.paste(_c14, (101, 2575), _c14)
except Exception:
    pass
layout["Free"] = [101, 2575, 235, 2681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 65, 65)
    canvas.paste(_c15, (308, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [308, 1, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/16_icon_IO_O0_AM.png
try:
    _c16 = get_crop(16, 293, 144)
    canvas.paste(_c16, (144, 1250), _c16)
except Exception:
    pass
layout["IO:O0_AM"] = [144, 1250, 437, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 57, 66)
    canvas.paste(_c17, (1317, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1317, 0, 1374, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/18_text_5.10_my.png
try:
    _c18 = get_crop(18, 149, 45)
    canvas.paste(_c18, (22, 15), _c18)
except Exception:
    pass
layout["5.10_my"] = [22, 15, 171, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/19_text_Cole_Gardens.png
try:
    _c19 = get_crop(19, 293, 144)
    canvas.paste(_c19, (144, 1250), _c19)
except Exception:
    pass
layout["Cole_Gardens"] = [144, 1250, 437, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/20_text_10_Followers.png
try:
    _c20 = get_crop(20, 293, 144)
    canvas.paste(_c20, (144, 1250), _c20)
except Exception:
    pass
layout["10_Followers"] = [144, 1250, 437, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/21_text_About_this_event.png
try:
    _c21 = get_crop(21, 454, 61)
    canvas.paste(_c21, (45, 2080), _c21)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 499, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/22_text_Hobbies_Special_Interest.png
try:
    _c22 = get_crop(22, 496, 55)
    canvas.paste(_c22, (86, 2192), _c22)
except Exception:
    pass
layout["Hobbies_&_Special_Interes"] = [86, 2192, 582, 2247]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/23_text_Anime_Comics.png
try:
    _c23 = get_crop(23, 288, 50)
    canvas.paste(_c23, (597, 2192), _c23)
except Exception:
    pass
layout["Anime_Comics"] = [597, 2192, 885, 2242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_08_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-10/24_text_General_Admission.png
try:
    _c24 = get_crop(24, 75, 72)
    canvas.paste(_c24, (249, 2588), _c24)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]
