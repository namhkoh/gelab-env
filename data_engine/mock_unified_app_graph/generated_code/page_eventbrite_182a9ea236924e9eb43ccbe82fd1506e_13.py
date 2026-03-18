# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_13
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15.png
# step_index: 13/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for the Eventbrite-like page.
# Uses provided canvas (PIL Image) and draw (PIL ImageDraw).
# Do not draw any detected icons/text/buttons — only backgrounds, banners, dividers, and structural cards.

# Clear/fill background (dominant color is white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~50px) - dark muted gray
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(154, 162, 162))  # #9AA2A2-ish

# Success banner under the status bar (mint/green feedback background)
banner_y0 = status_h
banner_y1 = 220
mint_green = (232, 248, 240)  # soft mint
draw.rectangle([(0, banner_y0), (1440, banner_y1)], fill=mint_green)

# thin divider under the banner
draw.line([(24, banner_y1), (1440-24, banner_y1)], fill=(226, 230, 230), width=1)

# Major content section separators (light subtle lines)
separator_color = (238, 238, 240)
# Under the top info block (around refund/policy area)
draw.line([(24, 560), (1440-24, 560)], fill=separator_color, width=2)
# Divider after "About this event" block
draw.line([(24, 1080), (1440-24, 1080)], fill=separator_color, width=2)
# Divider after Location block
draw.line([(24, 1620), (1440-24, 1620)], fill=separator_color, width=2)

# Soft section background for the "About this event" area (keeps content visually grouped)
about_y0 = 760
about_y1 = 1060
about_bg = (255, 255, 255)  # keep white but draw a very subtle overlay to separate sections
draw.rectangle([(24, about_y0), (1440-24, about_y1)], fill=about_bg)

# Light pill-like background examples (do not draw actual pills/icons/text).
# We won't draw any pills that match detected elements; just give subtle rounded card areas for grouping.
# Small subtle rounded rectangle behind meta area near top (location/4hrs/refund rows region)
meta_y0 = 240
meta_y1 = 540
draw.rounded_rectangle([(24, meta_y0), (1440-24, meta_y1)], radius=8, outline=(245,245,246), width=1, fill=(255,255,255))

# Large rounded card container for ticket selection above the reserve button
card_x0 = 48
card_x1 = 1440-48
card_y0 = 2320
card_y1 = 2680
card_radius = 20
card_border_color = (45, 86, 255)  # vivid bluish border
card_border_width = 6

# Outer border (rounded)
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)],
                       radius=card_radius, fill=None, outline=card_border_color, width=card_border_width)
# Inner fill to ensure crisp white interior
inner_inset = card_border_width + 6
draw.rounded_rectangle([(card_x0+inner_inset, card_y0+inner_inset),
                        (card_x1-inner_inset, card_y1-inner_inset)],
                       radius=card_radius-8, fill=(255,255,255), outline=None)

# subtle shadow line above the card to lift it visually
draw.line([(card_x0, card_y0-8), (card_x1, card_y0-8)], fill=(230,230,235), width=3)

# Light horizontal rule separating ticket title area and ticket price area inside the card
inner_sep_y = card_y0 + 120
draw.line([(card_x0+20, inner_sep_y), (card_x1-20, inner_sep_y)], fill=(245,246,250), width=1)

# Provide subtle dividing margin above the bottom Reserve area (so auto-pasted button sits on white)
reserve_top = 2736
draw.line([(24, reserve_top), (1440-24, reserve_top)], fill=(240,240,242), width=2)

# Small bottom safe-area shadow to separate reserve button region from content above
draw.rectangle([(0, 2720), (1440, 2960)], fill=(255,255,255,))  # keep white base
draw.line([(0, 2720), (1440, 2720)], fill=(230,230,235), width=2)

# Subtle large section cue: put a faint circular avatar placeholder ring center (structure-only, ensure NOT to draw image)
# We'll draw only a faint ring (no face), but avoid the exact detected avatar area if it matches detected icons.
# The detected avatar-like crop will be pasted on top; we opt NOT to draw any circle here to avoid duplication.

# Top toolbar subtle bottom border (below status elements)
draw.line([(0, status_h), (1440, status_h)], fill=(200,200,200), width=1)

# Right-side subtle "Show map" area divider (visual anchor) - just a faint guideline line
draw.line([(1160, 1460), (1440-24, 1460)], fill=(245,245,246), width=1)

# End of structural drawing.
# (No text, icons, or buttons drawn — those will be pasted later by the detection pipeline.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/00_icon_Job_Seekers.png
try:
    _c0 = get_crop(0, 240, 240)
    canvas.paste(_c0, (600, 1990), _c0)
except Exception:
    pass
layout["Job_Seekers"] = [600, 1990, 840, 2230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/02_icon_Increase.png
try:
    _c2 = get_crop(2, 96, 96)
    canvas.paste(_c2, (1224, 2444), _c2)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 94, 103)
    canvas.paste(_c3, (1107, 2441), _c3)
except Exception:
    pass
layout["icon_3"] = [1107, 2441, 1201, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/04_icon_4hrs.png
try:
    _c4 = get_crop(4, 199, 76)
    canvas.paste(_c4, (50, 469), _c4)
except Exception:
    pass
layout["4hrs"] = [50, 469, 249, 545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 53, 58)
    canvas.paste(_c5, (315, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [315, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/06_icon_Reserve_a_spot.png
try:
    _c6 = get_crop(6, 1296, 132)
    canvas.paste(_c6, (72, 2756), _c6)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/07_icon_Business_Professional.png
try:
    _c7 = get_crop(7, 234, 144)
    canvas.paste(_c7, (48, 1235), _c7)
except Exception:
    pass
layout["Business_&_Professional"] = [48, 1235, 282, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/08_icon_9.33.png
try:
    _c8 = get_crop(8, 53, 59)
    canvas.paste(_c8, (116, 4), _c8)
except Exception:
    pass
layout["9.33"] = [116, 4, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/09_icon_9.33.png
try:
    _c9 = get_crop(9, 57, 59)
    canvas.paste(_c9, (179, 3), _c9)
except Exception:
    pass
layout["9.33"] = [179, 3, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/10_icon_Share.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1260, 108), _c10)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 55, 58)
    canvas.paste(_c11, (246, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 4, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 56)
    canvas.paste(_c12, (1319, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 4, 1371, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 69, 62)
    canvas.paste(_c13, (1212, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 1, 1281, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/14_icon_Show_map.png
try:
    _c14 = get_crop(14, 226, 144)
    canvas.paste(_c14, (1166, 1453), _c14)
except Exception:
    pass
layout["Show_map"] = [1166, 1453, 1392, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/15_icon_New_York.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 325), _c15)
except Exception:
    pass
layout["New_York"] = [48, 325, 1392, 469]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/16_icon_you_looking_for_ajob_in_New_York_If_you_.png
try:
    _c16 = get_crop(16, 234, 144)
    canvas.paste(_c16, (48, 1235), _c16)
except Exception:
    pass
layout["you_looking_for_ajob_in_N"] = [48, 1235, 282, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/17_icon_Free.png
try:
    _c17 = get_crop(17, 75, 72)
    canvas.paste(_c17, (249, 2588), _c17)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/18_icon_Free.png
try:
    _c18 = get_crop(18, 139, 108)
    canvas.paste(_c18, (96, 2571), _c18)
except Exception:
    pass
layout["Free"] = [96, 2571, 235, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 42, 57)
    canvas.paste(_c19, (1272, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [1272, 4, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/20_icon_New_York_Virtual_Job_Fair_New_York_NY_10.png
try:
    _c20 = get_crop(20, 226, 144)
    canvas.paste(_c20, (1166, 1453), _c20)
except Exception:
    pass
layout["New_York;_Virtual_Job_Fai"] = [1166, 1453, 1392, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/21_icon_9.33.png
try:
    _c21 = get_crop(21, 91, 58)
    canvas.paste(_c21, (16, 5), _c21)
except Exception:
    pass
layout["9.33"] = [16, 5, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/22_icon_The_organizer_will_review_refund_request.png
try:
    _c22 = get_crop(22, 1344, 144)
    canvas.paste(_c22, (48, 325), _c22)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 325, 1392, 469]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 49, 61)
    canvas.paste(_c23, (383, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [383, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/24_icon_9.33.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (36, 108), _c24)
except Exception:
    pass
layout["9.33"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/25_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c25 = get_crop(25, 1440, 312)
    canvas.paste(_c25, (0, 0), _c25)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/26_text_Location.png
try:
    _c26 = get_crop(26, 246, 63)
    canvas.paste(_c26, (41, 1498), _c26)
except Exception:
    pass
layout["Location"] = [41, 1498, 287, 1561]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/27_text_General_Admission.png
try:
    _c27 = get_crop(27, 75, 72)
    canvas.paste(_c27, (249, 2588), _c27)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_13_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-15/28_text_Job_Seekers.png
try:
    _c28 = get_crop(28, 279, 52)
    canvas.paste(_c28, (562, 2451), _c28)
except Exception:
    pass
layout["Job_Seekers"] = [562, 2451, 841, 2503]
