# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_11
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13.png
# step_index: 11/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level UI background & structure drawing for the provided canvas
# Uses provided variables: canvas (PIL Image) and draw (ImageDraw.Draw)
# Assumes font_sm, font_md, font_lg, font_xl variables are available but not used for text.

w, h = canvas.size

# 1) Overall background (dominant very light neutral from screenshot)
draw.rectangle((0, 0, w, h), fill="#fbfbfd")

# 2) Status bar area at top (~72px high)
status_h = 72
draw.rectangle((0, 0, w, status_h), fill="#d6d6d6")
# subtle bottom divider of status bar
draw.line((0, status_h-1, w, status_h-1), fill="#bfbfbf", width=1)

# 3) Large header / hero image background (blurred image area)
hero_top = status_h
hero_bottom = 660
# vertical gradient (dark -> lighter) to emulate the blurred banner background
for i in range(hero_top, hero_bottom):
    t = (i - hero_top) / max(1, (hero_bottom - hero_top - 1))
    # color gradient from muted dark teal-gray to soft warm gray
    r = int(92 + (245 - 92) * t)   # 92 -> 245
    g = int(102 + (245 - 102) * t) # 102 -> 245
    b = int(104 + (250 - 104) * t) # 104 -> 250
    draw.line((0, i, w, i), fill=(r, g, b))

# subtle dark overlay band near the lower portion of the hero (to match screenshot's vignette)
overlay_top = hero_bottom - 120
overlay_bottom = hero_bottom - 40
for i in range(overlay_top, overlay_bottom):
    t = (i - overlay_top) / max(1, overlay_bottom - overlay_top - 1)
    alpha = int(48 * (1 - t))  # fading overlay
    # blend onto existing (approximate by drawing semi-opaque gray lines)
    gray = int(30 + 40 * (1 - t))
    draw.line((0, i, w, i), fill=(gray, gray, gray))

# 4) Image progress bar / photo indicator near bottom of hero area (thin bars)
pb_y = hero_bottom - 28
# long faint track
draw.rectangle((60, pb_y, w - 60, pb_y + 6), fill="#e9e9ea")
# shorter active segment
active_w = int((w - 120) * 0.42)
draw.rectangle((60, pb_y, 60 + active_w, pb_y + 6), fill="#ffffff")

# 5) Main page divider under hero
draw.line((40, hero_bottom + 8, w - 40, hero_bottom + 8), fill="#ece9ef", width=2)

# 6) Organizer / follow card background (rounded rectangle)
card_pad_x = 40
card_top = 1200
card_bottom = 1320
card_radius = 28
draw.rounded_rectangle((card_pad_x, card_top, w - card_pad_x, card_bottom),
                       radius=card_radius, fill="#f6f4f8", outline=None)

# subtle inner horizontal divider in the card (to separate organizer name and followers line)
divider_y = card_top + 56
draw.line((card_pad_x + 24, divider_y, w - card_pad_x - 180, divider_y), fill="#edeaf0", width=1)

# 7) Separator and info area dividers for event details
# thin divider below the location/refund policy area (approx)
sep1_y = 1500
draw.line((40, sep1_y, w - 40, sep1_y), fill="#efecef", width=1)

# another subtle divider between content blocks
sep2_y = 1900
draw.line((40, sep2_y, w - 40, sep2_y), fill="#f0eef3", width=1)

# 8) "About this event" card background area (just a subtle white panel region)
about_top = 2040
about_bottom = 2340
draw.rectangle((40, about_top, w - 40, about_bottom), fill="#ffffff")
# light border to lift the area from the page
draw.rectangle((40, about_top, w - 40, about_bottom), outline="#f0edf3", width=1)

# 9) Bottom location / map area card (rounded rectangle)
loc_top = 2580
loc_bottom = 2890
loc_radius = 20
draw.rounded_rectangle((40, loc_top, w - 40, loc_bottom), radius=loc_radius, fill="#ffffff", outline="#f1eef4")

# 10) Subtle shadow accents underneath major panels to separate layers
# shadow under organizer card
shadow_top = card_bottom + 6
draw.line((card_pad_x + 8, shadow_top, w - card_pad_x - 8, shadow_top), fill="#efe9f2", width=2)
# shadow under about panel
draw.line((42, about_bottom + 6, w - 42, about_bottom + 6), fill="#f6f5f8", width=2)
# shadow under location card
draw.line((42, loc_bottom + 6, w - 42, loc_bottom + 6), fill="#f6f5f8", width=2)

# 11) Small decorative left accent for "About this event" heading area (no text drawn)
accent_x = 40
accent_w = 6
draw.rectangle((accent_x, about_top + 16, accent_x + accent_w, about_top + 72), fill="#efe5ff")

# 12) Ensure there is breathing space: faint horizontal rule above location card
draw.line((40, loc_top - 24, w - 40, loc_top - 24), fill="#f0edf3", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/01_icon_Going_fast.png
try:
    _c1 = get_crop(1, 334, 85)
    canvas.paste(_c1, (41, 753), _c1)
except Exception:
    pass
layout["Going_fast"] = [41, 753, 375, 838]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/02_icon_yFiTNesS.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["@yFiTNesS"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/03_icon_7.55.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["7.55"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/04_icon_Sports_Fitness.png
try:
    _c4 = get_crop(4, 234, 144)
    canvas.paste(_c4, (48, 2427), _c4)
except Exception:
    pass
layout["Sports_&_Fitness"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/05_icon_Share.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1260, 108), _c5)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/06_icon_Ticket_sales_end_soon.png
try:
    _c6 = get_crop(6, 547, 84)
    canvas.paste(_c6, (379, 753), _c6)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [379, 753, 926, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/07_icon_7.55.png
try:
    _c7 = get_crop(7, 65, 67)
    canvas.paste(_c7, (178, 1), _c7)
except Exception:
    pass
layout["7.55"] = [178, 1, 243, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 62)
    canvas.paste(_c8, (1320, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1320, 2, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 99, 61)
    canvas.paste(_c9, (1215, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1215, 2, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 55, 64)
    canvas.paste(_c10, (247, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [247, 1, 302, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 64, 63)
    canvas.paste(_c11, (308, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [308, 2, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/12_icon_7.55.png
try:
    _c12 = get_crop(12, 61, 68)
    canvas.paste(_c12, (114, 0), _c12)
except Exception:
    pass
layout["7.55"] = [114, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/13_icon_Niniasa_Flow.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1116, 108), _c13)
except Exception:
    pass
layout["Niniasa_Flow"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 50, 63)
    canvas.paste(_c14, (383, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [383, 2, 433, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/15_text_7.55.png
try:
    _c15 = get_crop(15, 92, 43)
    canvas.paste(_c15, (22, 17), _c15)
except Exception:
    pass
layout["7.55"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/16_text_yFiTNesS.png
try:
    _c16 = get_crop(16, 221, 55)
    canvas.paste(_c16, (784, 101), _c16)
except Exception:
    pass
layout["@yFiTNesS"] = [784, 101, 1005, 156]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/17_text_Sunday_April_28.png
try:
    _c17 = get_crop(17, 416, 77)
    canvas.paste(_c17, (38, 886), _c17)
except Exception:
    pass
layout["Sunday;_April_28"] = [38, 886, 454, 963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/18_text_9_30AM.png
try:
    _c18 = get_crop(18, 212, 56)
    canvas.paste(_c18, (483, 893), _c18)
except Exception:
    pass
layout["9:30AM"] = [483, 893, 695, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/19_text_Lululemon_X_Nu_Fitness_Vinyasa_Yoga.png
try:
    _c19 = get_crop(19, 225, 144)
    canvas.paste(_c19, (144, 1250), _c19)
except Exception:
    pass
layout["Lululemon_X_Nu_Fitness_Vi"] = [144, 1250, 369, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/20_text_Class.png
try:
    _c20 = get_crop(20, 199, 72)
    canvas.paste(_c20, (42, 1113), _c20)
except Exception:
    pass
layout["Class"] = [42, 1113, 241, 1185]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/21_text_Nu_Fitness.png
try:
    _c21 = get_crop(21, 225, 144)
    canvas.paste(_c21, (144, 1250), _c21)
except Exception:
    pass
layout["Nu_Fitness"] = [144, 1250, 369, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/22_text_10_Followers.png
try:
    _c22 = get_crop(22, 225, 144)
    canvas.paste(_c22, (144, 1250), _c22)
except Exception:
    pass
layout["10_Followers"] = [144, 1250, 369, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/23_text_lululemon_Broadway_Oakland_CA_USA.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1517), _c23)
except Exception:
    pass
layout["lululemon;_Broadway;_Oakl"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/24_text_hrs.png
try:
    _c24 = get_crop(24, 77, 50)
    canvas.paste(_c24, (176, 1674), _c24)
except Exception:
    pass
layout["hrs"] = [176, 1674, 253, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1780), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/26_text_The_organizer_will_review_refund_request.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1517), _c26)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/27_text_About_this_event.png
try:
    _c27 = get_crop(27, 454, 61)
    canvas.paste(_c27, (45, 2080), _c27)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 499, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/28_text_Join_us_for_an_alignment-based_hatha_vin.png
try:
    _c28 = get_crop(28, 234, 144)
    canvas.paste(_c28, (48, 2427), _c28)
except Exception:
    pass
layout["Join_us_for_an_alignment-"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/29_text_Read_more.png
try:
    _c29 = get_crop(29, 234, 144)
    canvas.paste(_c29, (48, 2427), _c29)
except Exception:
    pass
layout["Read_more"] = [48, 2427, 282, 2571]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/30_text_Location.png
try:
    _c30 = get_crop(30, 244, 61)
    canvas.paste(_c30, (43, 2691), _c30)
except Exception:
    pass
layout["Location"] = [43, 2691, 287, 2752]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/31_text_Show_map.png
try:
    _c31 = get_crop(31, 226, 144)
    canvas.paste(_c31, (1166, 2645), _c31)
except Exception:
    pass
layout["Show_map"] = [1166, 2645, 1392, 2789]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_11_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-13/32_text_lululemon_Broadway_Oakland_CA_USA.png
try:
    _c32 = get_crop(32, 234, 144)
    canvas.paste(_c32, (48, 2427), _c32)
except Exception:
    pass
layout["lululemon,_Broadway;_Oakl"] = [48, 2427, 282, 2571]
