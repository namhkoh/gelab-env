# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_09
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11.png
# step_index: 9/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for the mobile event page
w, h = canvas.size

# 1) Overall page background (very light off-white to match app background)
draw.rectangle([(0, 0), (w, h)], fill="#FBFAFD")  # subtle off-white

# 2) Status bar area (top ~50px) — slightly darker gray strip
status_h = 50
draw.rectangle([(0, 0), (w, status_h)], fill="#CFCFCF")

# 3) Hero/header image area placeholder (dark band beneath status bar)
hero_top = status_h
hero_bottom = 480
# simple vertical gradient simulation using horizontal lines
top_color = (28, 28, 28)    # dark
bottom_color = (60, 48, 46) # slightly warmer dark
height = hero_bottom - hero_top
for i in range(height):
    t = i / max(1, height - 1)
    r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
    g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
    b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
    draw.line([(0, hero_top + i), (w, hero_top + i)], fill=(r, g, b))

# subtle rounded mask on hero (gives softer bottom edge)
mask_radius = 20
draw.rounded_rectangle([(0, hero_top), (w, hero_bottom)], radius=0, outline=None)

# 4) Soft overlay vignette on hero edges (darker corners) - drawn as translucent-ish rectangles
# (Use darker stripes at left/right to emulate vignette)
vignette_width = 220
draw.rectangle([(0, hero_top), (vignette_width, hero_bottom)], fill=(18, 18, 18, 40))
draw.rectangle([(w - vignette_width, hero_top), (w, hero_bottom)], fill=(18, 18, 18, 40))

# 5) Main content area background (white card area begins below hero)
content_top = hero_bottom + 20
draw.rectangle([(0, content_top), (w, h)], fill="#FFFFFF")

# 6) Organizer/info card background (rounded rectangle behind organizer row)
card_x1, card_x2 = 48, w - 48
card_y1, card_y2 = 1020, 1280  # sits under the event title area
draw.rounded_rectangle([(card_x1, card_y1), (card_x2, card_y2)], radius=24, fill="#F6F5F8", outline="#ECE8F1")

# 7) Thin divider line below organizer card
divider_y = card_y2 + 40
draw.line([(48, divider_y), (w - 48, divider_y)], fill="#E9E6EE", width=2)

# 8) Info list area (icons + lines area) background remains white; add subtle separators for rows
info_start_y = divider_y + 30
row_height = 86
for i in range(3):
    y = info_start_y + i * (row_height + 12)
    # draw subtle separator line between rows (except first)
    if i > 0:
        draw.line([(48, y - 10), (w - 48, y - 10)], fill="#F0EDF4", width=1)

# 9) Larger horizontal divider before "About this event"
about_divider_y = info_start_y + 3 * (row_height + 12) + 10
draw.line([(48, about_divider_y), (w - 48, about_divider_y)], fill="#E7E3EA", width=2)

# 10) About section background area (slightly tinted to separate)
about_top = about_divider_y + 28
about_bottom = about_top + 420
draw.rectangle([(48, about_top), (w - 48, about_bottom)], fill="#FFFFFF")  # keep white but separated by spacing

# thin inner top border for the About section to emphasize heading separation
draw.line([(48, about_top), (w - 48, about_top)], fill="#F0EDF4", width=1)

# 11) Light bottom separator further down to structure page
lower_divider_y = about_bottom + 60
draw.line([(24, lower_divider_y), (w - 24, lower_divider_y)], fill="#F1EFF4", width=2)

# 12) Decorative circular watermark area at lower center (subtle, behind profile area)
circle_center_x = w // 2
circle_center_y = lower_divider_y + 360
circle_r = 120
# very light ring
draw.ellipse([(circle_center_x - circle_r, circle_center_y - circle_r),
              (circle_center_x + circle_r, circle_center_y + circle_r)],
             outline="#F0EFFF", width=6)

# 13) Final subtle bottom background band to indicate page continuation
band_top = h - 220
draw.rectangle([(0, band_top), (w, h)], fill="#FBFAFD")
draw.line([(48, band_top), (w - 48, band_top)], fill="#EEEAF2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/01_icon_Fashion_Beauty.png
try:
    _c1 = get_crop(1, 234, 144)
    canvas.paste(_c1, (48, 2427), _c1)
except Exception:
    pass
layout["Fashion_&_Beauty"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/02_icon_Going_fast.png
try:
    _c2 = get_crop(2, 332, 86)
    canvas.paste(_c2, (42, 753), _c2)
except Exception:
    pass
layout["Going_fast"] = [42, 753, 374, 839]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/03_icon_REFER.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["REFER"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/04_icon_5.28.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["5.28"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/05_icon_REFER.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["REFER"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/06_icon_Ticket_sales_end_soon.png
try:
    _c6 = get_crop(6, 546, 84)
    canvas.paste(_c6, (379, 753), _c6)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [379, 753, 925, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/07_icon_5.28.png
try:
    _c7 = get_crop(7, 61, 68)
    canvas.paste(_c7, (181, 1), _c7)
except Exception:
    pass
layout["5.28"] = [181, 1, 242, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 97, 62)
    canvas.paste(_c8, (1215, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1215, 1, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 63)
    canvas.paste(_c9, (1317, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1317, 1, 1373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/10_icon_5.28.png
try:
    _c10 = get_crop(10, 62, 69)
    canvas.paste(_c10, (114, 0), _c10)
except Exception:
    pass
layout["5.28"] = [114, 0, 176, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/11_icon_F_REE.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1116, 108), _c11)
except Exception:
    pass
layout["F_REE"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 55, 68)
    canvas.paste(_c12, (247, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 0, 302, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 68, 67)
    canvas.paste(_c13, (307, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [307, 0, 375, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/14_text_5.28.png
try:
    _c14 = get_crop(14, 92, 43)
    canvas.paste(_c14, (22, 17), _c14)
except Exception:
    pass
layout["5.28"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/15_text_Monday_May_6_._10.00_AM.png
try:
    _c15 = get_crop(15, 340, 144)
    canvas.paste(_c15, (288, 1250), _c15)
except Exception:
    pass
layout["Monday;_May_6_._10.00_AM"] = [288, 1250, 628, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/16_text_Hair_3R_s.png
try:
    _c16 = get_crop(16, 311, 74)
    canvas.paste(_c16, (44, 1016), _c16)
except Exception:
    pass
layout["Hair_3R's"] = [44, 1016, 355, 1090]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/17_text_Recognise_Respond_Refer.png
try:
    _c17 = get_crop(17, 331, 144)
    canvas.paste(_c17, (1013, 1290), _c17)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/18_text_Online_Professional_Development.png
try:
    _c18 = get_crop(18, 340, 144)
    canvas.paste(_c18, (288, 1250), _c18)
except Exception:
    pass
layout["Online_Professional_Devel"] = [288, 1250, 628, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/19_text_of_Kingston.png
try:
    _c19 = get_crop(19, 340, 144)
    canvas.paste(_c19, (288, 1250), _c19)
except Exception:
    pass
layout["of_Kingston"] = [288, 1250, 628, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/20_text_291_Followers.png
try:
    _c20 = get_crop(20, 340, 144)
    canvas.paste(_c20, (288, 1250), _c20)
except Exception:
    pass
layout["291_Followers"] = [288, 1250, 628, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/21_text_Online_event.png
try:
    _c21 = get_crop(21, 275, 55)
    canvas.paste(_c21, (138, 1563), _c21)
except Exception:
    pass
layout["Online_event"] = [138, 1563, 413, 1618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/22_text_2_hrs_30_mins.png
try:
    _c22 = get_crop(22, 290, 54)
    canvas.paste(_c22, (141, 1672), _c22)
except Exception:
    pass
layout["2_hrs_30_mins"] = [141, 1672, 431, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 63)
    canvas.paste(_c23, (138, 1780), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1517), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/25_text_About_this_event.png
try:
    _c25 = get_crop(25, 452, 57)
    canvas.paste(_c25, (46, 2081), _c25)
except Exception:
    pass
layout["About_this_event"] = [46, 2081, 498, 2138]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/26_text_Online_training_session_for_hairdressers.png
try:
    _c26 = get_crop(26, 240, 187)
    canvas.paste(_c26, (600, 2773), _c26)
except Exception:
    pass
layout["Online_training_session_f"] = [600, 2773, 840, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/27_text_Read_more.png
try:
    _c27 = get_crop(27, 234, 144)
    canvas.paste(_c27, (48, 2427), _c27)
except Exception:
    pass
layout["Read_more"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_09_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-11/28_clickable_Organizer_profile_picture.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (96, 1289), _c28)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1289, 240, 1433]
