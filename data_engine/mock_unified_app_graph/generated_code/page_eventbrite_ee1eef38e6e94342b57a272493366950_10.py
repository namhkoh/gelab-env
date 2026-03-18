# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_10
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12.png
# step_index: 10/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (slightly warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFC")

# Top status bar (do not draw icons/text inside it)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#9DA39B")  # muted grey-green status bar

# Notification banner under status bar (rounded rect)
notif_top = status_h + 12
notif_bottom = notif_top + 88
draw.rounded_rectangle(
    [(24, notif_top), (1440 - 24, notif_bottom)],
    radius=8,
    fill="#EAF6EE",  # pale green banner
    outline=None
)

# Decorative thin divider under notification area
draw.line([(24, notif_bottom + 12), (1440 - 24, notif_bottom + 12)], fill="#E6E6E6", width=1)

# Large header image/banner placeholder (dark blurred strip)
img_top = notif_bottom + 24
img_bottom = img_top + 320
# simple vertical gradient approximation
for i in range(img_bottom - img_top):
    t = i / max(1, (img_bottom - img_top - 1))
    # gradient from warm dark to slightly lighter
    r = int(40 * (1 - t) + 20 * t)
    g = int(30 * (1 - t) + 16 * t)
    b = int(30 * (1 - t) + 12 * t)
    draw.line([(0, img_top + i), (1440, img_top + i)], fill=(r, g, b))

# Subtle bottom fade line under image
draw.line([(24, img_bottom + 6), (1440 - 24, img_bottom + 6)], fill="#EDEDED", width=1)

# Organizer/follow card background (rounded rectangle behind profile + follow button)
card_x0 = 40
card_x1 = 1440 - 40
card_y0 = 1188
card_y1 = card_y0 + 160
draw.rounded_rectangle(
    [(card_x0, card_y0), (card_x1, card_y1)],
    radius=28,
    fill="#F6F5F8",       # very light neutral card color
    outline="#EAE8EE"     # subtle border
)

# Small shadow under card
draw.line([(card_x0 + 6, card_y1 + 4), (card_x1 - 6, card_y1 + 4)], fill="#F0EFF2", width=2)

# Section separators between content blocks
sep1_y = 1500
sep2_y = 2028
draw.line([(32, sep1_y), (1440 - 32, sep1_y)], fill="#ECE9EE", width=2)
draw.line([(32, sep2_y), (1440 - 32, sep2_y)], fill="#ECE9EE", width=2)

# "About this event" header background hint (very light, so it doesn't duplicate detected text)
about_bg_top = sep2_y + 18
about_bg_bottom = about_bg_top + 90
draw.rectangle([(32, about_bg_top), (1440 - 32, about_bg_bottom)], fill="#FFFFFF")  # keep white but define area

# Light rounded card behind the small category/tag area (do not draw the tag itself)
tag_bg_x0 = 40
tag_bg_y0 = about_bg_top + 48
tag_bg_x1 = tag_bg_x0 + 420
tag_bg_y1 = tag_bg_y0 + 54
draw.rounded_rectangle(
    [(tag_bg_x0, tag_bg_y0), (tag_bg_x1, tag_bg_y1)],
    radius=28,
    fill="#F3F4F6",   # subtle neutral pill background
    outline=None
)

# Final subtle bottom padding divider before the Reserve area (do not draw the reserve button)
bottom_div_y = 2320
draw.line([(24, bottom_div_y), (1440 - 24, bottom_div_y)], fill="#E9E7EA", width=2)

# Add a faint left accent stripe to visually separate content (purely decorative)
draw.rectangle([(32, img_bottom + 18), (40, bottom_div_y - 18)], fill="#F0EEF4")

# End of background/structure drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/01_icon_Going_fast.png
try:
    _c1 = get_crop(1, 332, 85)
    canvas.paste(_c1, (42, 754), _c1)
except Exception:
    pass
layout["Going_fast"] = [42, 754, 374, 839]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 113, 106)
    canvas.paste(_c2, (987, 2439), _c2)
except Exception:
    pass
layout["icon_2"] = [987, 2439, 1100, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 106, 103)
    canvas.paste(_c3, (1217, 2441), _c3)
except Exception:
    pass
layout["icon_3"] = [1217, 2441, 1323, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1440, 636)
    canvas.paste(_c4, (0, 2324), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 94, 104)
    canvas.paste(_c5, (1108, 2440), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2440, 1202, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/06_icon_Fashion_Beauty.png
try:
    _c6 = get_crop(6, 585, 98)
    canvas.paste(_c6, (40, 2167), _c6)
except Exception:
    pass
layout["Fashion_&_Beauty"] = [40, 2167, 625, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/07_icon_REFER.png
try:
    _c7 = get_crop(7, 142, 142)
    canvas.paste(_c7, (1251, 97), _c7)
except Exception:
    pass
layout["REFER"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/08_icon_5.28.png
try:
    _c8 = get_crop(8, 64, 64)
    canvas.paste(_c8, (178, 1), _c8)
except Exception:
    pass
layout["5.28"] = [178, 1, 242, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 59, 59)
    canvas.paste(_c9, (311, 4), _c9)
except Exception:
    pass
layout["icon_9"] = [311, 4, 370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/10_icon_5.28.png
try:
    _c10 = get_crop(10, 63, 65)
    canvas.paste(_c10, (113, 0), _c10)
except Exception:
    pass
layout["5.28"] = [113, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 49, 54)
    canvas.paste(_c11, (250, 7), _c11)
except Exception:
    pass
layout["icon_11"] = [250, 7, 299, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 56, 63)
    canvas.paste(_c12, (1317, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1317, 1, 1373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 80, 64)
    canvas.paste(_c13, (1211, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1211, 1, 1291, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/14_icon_of_Kingston.png
try:
    _c14 = get_crop(14, 340, 144)
    canvas.paste(_c14, (288, 1250), _c14)
except Exception:
    pass
layout["of_Kingston"] = [288, 1250, 628, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 64)
    canvas.paste(_c15, (382, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 2, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/16_icon_Ticket_sales_end_soon.png
try:
    _c16 = get_crop(16, 547, 83)
    canvas.paste(_c16, (379, 753), _c16)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [379, 753, 926, 836]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 42, 59)
    canvas.paste(_c17, (1272, 4), _c17)
except Exception:
    pass
layout["icon_17"] = [1272, 4, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/18_icon_5.28.png
try:
    _c18 = get_crop(18, 92, 62)
    canvas.paste(_c18, (15, 2), _c18)
except Exception:
    pass
layout["5.28"] = [15, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/19_icon_F_REE.png
try:
    _c19 = get_crop(19, 142, 142)
    canvas.paste(_c19, (1251, 97), _c19)
except Exception:
    pass
layout["F_REE"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/20_icon_Free.png
try:
    _c20 = get_crop(20, 134, 105)
    canvas.paste(_c20, (101, 2576), _c20)
except Exception:
    pass
layout["Free"] = [101, 2576, 235, 2681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/21_icon_2_hrs_30_mins.png
try:
    _c21 = get_crop(21, 374, 70)
    canvas.paste(_c21, (56, 1664), _c21)
except Exception:
    pass
layout["2_hrs_30_mins"] = [56, 1664, 430, 1734]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/22_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (36, 108), _c22)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/23_text_Monday_May_6_._10.00_AM.png
try:
    _c23 = get_crop(23, 340, 144)
    canvas.paste(_c23, (288, 1250), _c23)
except Exception:
    pass
layout["Monday;_May_6_._10.00_AM"] = [288, 1250, 628, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/24_text_Hair_3R_s.png
try:
    _c24 = get_crop(24, 311, 74)
    canvas.paste(_c24, (44, 1016), _c24)
except Exception:
    pass
layout["Hair_3R's"] = [44, 1016, 355, 1090]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/25_text_Recognise_Respond_Refer.png
try:
    _c25 = get_crop(25, 331, 144)
    canvas.paste(_c25, (1013, 1290), _c25)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/26_text_Online_Professional_Development.png
try:
    _c26 = get_crop(26, 340, 144)
    canvas.paste(_c26, (288, 1250), _c26)
except Exception:
    pass
layout["Online_Professional_Devel"] = [288, 1250, 628, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/27_text_Online_event.png
try:
    _c27 = get_crop(27, 275, 55)
    canvas.paste(_c27, (138, 1563), _c27)
except Exception:
    pass
layout["Online_event"] = [138, 1563, 413, 1618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/28_text_Refund_policy.png
try:
    _c28 = get_crop(28, 299, 63)
    canvas.paste(_c28, (138, 1780), _c28)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/29_text_The_organizer_will_review_refund_request.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1517), _c29)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/30_text_About_this_event.png
try:
    _c30 = get_crop(30, 452, 57)
    canvas.paste(_c30, (46, 2081), _c30)
except Exception:
    pass
layout["About_this_event"] = [46, 2081, 498, 2138]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/31_text_Online_access.png
try:
    _c31 = get_crop(31, 311, 52)
    canvas.paste(_c31, (116, 2451), _c31)
except Exception:
    pass
layout["Online_access"] = [116, 2451, 427, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_10_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-12/32_clickable_Organizer_profile_picture.png
try:
    _c32 = get_crop(32, 144, 144)
    canvas.paste(_c32, (96, 1289), _c32)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1289, 240, 1433]
