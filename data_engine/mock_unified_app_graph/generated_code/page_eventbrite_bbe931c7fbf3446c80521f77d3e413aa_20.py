# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_20
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22.png
# step_index: 20/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas/draw
# Assumes: canvas is a PIL.Image (1440x2960 RGB) and draw is ImageDraw.Draw(canvas)

# Colors
status_bar_color = (153, 169, 162)    # muted gray-green for status bar
notification_bg = (230, 245, 236)     # pale green notification banner
notif_border = (204, 230, 218)        # subtle border under notification
video_border_bg = (245, 245, 245)     # light border/background around video cards
video_fill = (48, 48, 48)             # dark filler for video placeholder areas
separator = (220, 223, 225)           # general separators
card_shadow = (230, 233, 236)         # subtle shadow under cards
card_fill = (255, 255, 255)           # white card background
card_border_blue = (44, 88, 255)      # blue border for ticket/card

width, height = canvas.size

# 1) Status bar at top (~96px)
status_bar_h = 96
draw.rectangle([0, 0, width, status_bar_h], fill=status_bar_color)

# 2) Notification banner under status bar (approx height to match detection area)
notif_y0 = status_bar_h
notif_y1 = 312  # matches detected banner height
draw.rectangle([0, notif_y0, width, notif_y1], fill=notification_bg)

# subtle divider under notification
draw.line([(32, notif_y1), (width-32, notif_y1)], fill=notif_border, width=2)

# 3) Video/content card placeholders (rounded rectangles)
# Detected large video thumbnails:
videos = [
    (58, 581, 1323, 755),   # (x, y, w, h) from detection [6]
    (58, 1357, 1323, 755)   # detection [18]
]
for (vx, vy, vw, vh) in videos:
    x0, y0 = vx, vy
    x1, y1 = vx + vw, vy + vh

    # light outer frame/background to mimic player card border
    outer = [x0-8, y0-8, x1+8, y1+8]
    try:
        draw.rounded_rectangle(outer, radius=14, fill=video_border_bg)
    except Exception:
        draw.rectangle(outer, fill=video_border_bg)

    # dark inner area where the video image will be pasted on top
    inner = [x0, y0, x1, y1]
    try:
        draw.rounded_rectangle(inner, radius=8, fill=video_fill)
    except Exception:
        draw.rectangle(inner, fill=video_fill)

    # subtle thin border line around the video card
    try:
        draw.rounded_rectangle(inner, radius=8, outline=separator, width=2)
    except Exception:
        draw.rectangle(inner, outline=separator, width=2)

# 4) Separator line after second video area
sep_y = videos[-1][1] + videos[-1][3] + 16
draw.line([(48, sep_y), (width-48, sep_y)], fill=separator, width=2)

# 5) Ticket / "Complimentary Access" card area
card_x0 = 48
card_x1 = width - 48
card_y0 = sep_y + 32
card_y1 = 2670  # leave space above the Reserve button (which will be pasted later)

# card shadow (slightly offset)
shadow_offset = 8
try:
    draw.rounded_rectangle([card_x0+shadow_offset, card_y0+shadow_offset,
                            card_x1+shadow_offset, card_y1+shadow_offset],
                           radius=20, fill=card_shadow)
except Exception:
    draw.rectangle([card_x0+shadow_offset, card_y0+shadow_offset, card_x1+shadow_offset, card_y1+shadow_offset],
                   fill=card_shadow)

# card background and blue border
try:
    draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1],
                           radius=20, fill=card_fill, outline=card_border_blue, width=6)
except Exception:
    draw.rectangle([card_x0, card_y0, card_x1, card_y1], fill=card_fill, outline=card_border_blue, width=6)

# inner divider line inside the card to suggest sections (do not draw any text)
inner_div_y = card_y0 + 84
draw.line([(card_x0+24, inner_div_y), (card_x1-24, inner_div_y)], fill=separator, width=1)

# small rounded container on the right where quantity controls will be placed (background only)
qty_box_w = 180
qty_box_h = 88
qty_x1 = card_x1 - 40
qty_x0 = qty_x1 - qty_box_w
qty_y0 = card_y0 + 28
qty_y1 = qty_y0 + qty_box_h
try:
    draw.rounded_rectangle([qty_x0, qty_y0, qty_x1, qty_y1], radius=14, fill=(245,245,250))
    draw.rounded_rectangle([qty_x0, qty_y0, qty_x1, qty_y1], radius=14, outline=(230,230,235), width=2)
except Exception:
    draw.rectangle([qty_x0, qty_y0, qty_x1, qty_y1], fill=(245,245,250), outline=(230,230,235), width=2)

# 6) Thin separators and subtle guide lines across the page
# top small divider under header area
draw.line([(32, notif_y1+8), (width-32, notif_y1+8)], fill=(240,240,240), width=1)

# faint horizontal rule near the bottom area above the reserve button
reserve_top_gap = 2756  # reserve button top (detect)
draw.line([(32, reserve_top_gap-40), (width-32, reserve_top_gap-40)], fill=(238,238,238), width=2)

# 7) Final subtle left/right gutters (vertical faint lines) to give structure
gutter_x = 40
draw.line([(gutter_x, notif_y1), (gutter_x, height-200)], fill=(250,250,250), width=1)
draw.line([(width-gutter_x, notif_y1), (width-gutter_x, height-200)], fill=(250,250,250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/00_icon_Decrease.png
try:
    _c0 = get_crop(0, 99, 96)
    canvas.paste(_c0, (996, 2444), _c0)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/01_icon_Empowering_Diversity_in_Franchising_How_.png
try:
    _c1 = get_crop(1, 1289, 20)
    canvas.paste(_c1, (75, 486), _c1)
except Exception:
    pass
layout["Empowering_Diversity_in_F"] = [75, 486, 1364, 506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 92, 103)
    canvas.paste(_c4, (1108, 2441), _c4)
except Exception:
    pass
layout["icon_4"] = [1108, 2441, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/05_icon_9.14.png
try:
    _c5 = get_crop(5, 51, 57)
    canvas.paste(_c5, (117, 4), _c5)
except Exception:
    pass
layout["9.14"] = [117, 4, 168, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/06_icon_Ae.png
try:
    _c6 = get_crop(6, 1323, 755)
    canvas.paste(_c6, (58, 581), _c6)
except Exception:
    pass
layout["Ae"] = [58, 581, 1381, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 64)
    canvas.paste(_c7, (1155, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1155, 2, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 56)
    canvas.paste(_c8, (184, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [184, 4, 234, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 57)
    canvas.paste(_c9, (315, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [315, 5, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 56)
    canvas.paste(_c10, (249, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [249, 5, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 42, 55)
    canvas.paste(_c11, (1328, 6), _c11)
except Exception:
    pass
layout["icon_11"] = [1328, 6, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 59, 59)
    canvas.paste(_c12, (1214, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1214, 3, 1273, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/13_icon_Share.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1260, 108), _c13)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/14_icon_Free.png
try:
    _c14 = get_crop(14, 134, 101)
    canvas.paste(_c14, (98, 2574), _c14)
except Exception:
    pass
layout["Free"] = [98, 2574, 232, 2675]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 40, 56)
    canvas.paste(_c15, (1273, 5), _c15)
except Exception:
    pass
layout["icon_15"] = [1273, 5, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 61)
    canvas.paste(_c16, (383, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [383, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/17_icon_Free.png
try:
    _c17 = get_crop(17, 75, 72)
    canvas.paste(_c17, (249, 2588), _c17)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/18_icon_Franchising_Learn_the_7_Habits_of_Highly.png
try:
    _c18 = get_crop(18, 1323, 755)
    canvas.paste(_c18, (58, 1357), _c18)
except Exception:
    pass
layout["Franchising:_Learn_the_7_"] = [58, 1357, 1381, 2112]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/19_icon_Share.png
try:
    _c19 = get_crop(19, 65, 84)
    canvas.paste(_c19, (1285, 581), _c19)
except Exception:
    pass
layout["Share"] = [1285, 581, 1350, 665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/20_icon_tion.png
try:
    _c20 = get_crop(20, 66, 66)
    canvas.paste(_c20, (1298, 494), _c20)
except Exception:
    pass
layout["tion"] = [1298, 494, 1364, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/21_icon_eadthfsl.png
try:
    _c21 = get_crop(21, 66, 66)
    canvas.paste(_c21, (75, 494), _c21)
except Exception:
    pass
layout["(eadthfsl"] = [75, 494, 141, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/22_text_9.14.png
try:
    _c22 = get_crop(22, 97, 49)
    canvas.paste(_c22, (18, 13), _c22)
except Exception:
    pass
layout["9.14"] = [18, 13, 115, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/23_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c23 = get_crop(23, 1440, 312)
    canvas.paste(_c23, (0, 0), _c23)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [0, 0, 1440, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/24_text_Read_less.png
try:
    _c24 = get_crop(24, 206, 144)
    canvas.paste(_c24, (48, 2131), _c24)
except Exception:
    pass
layout["Read_less"] = [48, 2131, 254, 2275]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/25_text_Complimentary_Access.png
try:
    _c25 = get_crop(25, 75, 72)
    canvas.paste(_c25, (249, 2588), _c25)
except Exception:
    pass
layout["Complimentary_Access"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/26_clickable_Back.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (36, 108), _c26)
except Exception:
    pass
layout["Back"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/27_clickable_Mute.png
try:
    _c27 = get_crop(27, 66, 66)
    canvas.paste(_c27, (141, 494), _c27)
except Exception:
    pass
layout["Mute"] = [141, 494, 207, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/28_clickable_0_00_1_09.png
try:
    _c28 = get_crop(28, 100, 66)
    canvas.paste(_c28, (207, 494), _c28)
except Exception:
    pass
layout["0:00___1:09"] = [207, 494, 307, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/29_clickable_Subtitles_closed_captions.png
try:
    _c29 = get_crop(29, 66, 66)
    canvas.paste(_c29, (1075, 494), _c29)
except Exception:
    pass
layout["Subtitles_closed_captions"] = [1075, 494, 1141, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/30_clickable_Settings.png
try:
    _c30 = get_crop(30, 65, 66)
    canvas.paste(_c30, (1141, 494), _c30)
except Exception:
    pass
layout["Settings"] = [1141, 494, 1206, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/31_clickable_Watch_on_YouTube.png
try:
    _c31 = get_crop(31, 92, 66)
    canvas.paste(_c31, (1206, 494), _c31)
except Exception:
    pass
layout["Watch_on_YouTube"] = [1206, 494, 1298, 560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/32_clickable_Photo_image_of_Sociallybuzz_Inc.png
try:
    _c32 = get_crop(32, 66, 66)
    canvas.paste(_c32, (68, 591), _c32)
except Exception:
    pass
layout["Photo_image_of_Sociallybu"] = [68, 591, 134, 657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/33_clickable_Empowering_Diversity_in_Franchising_How_.png
try:
    _c33 = get_crop(33, 1127, 33)
    canvas.paste(_c33, (144, 610), _c33)
except Exception:
    pass
layout["Empowering_Diversity_in_F"] = [144, 610, 1271, 643]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/34_clickable_Play.png
try:
    _c34 = get_crop(34, 93, 66)
    canvas.paste(_c34, (673, 926), _c34)
except Exception:
    pass
layout["Play"] = [673, 926, 766, 992]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/35_clickable_Watch_on_YouTube.png
try:
    _c35 = get_crop(35, 238, 65)
    canvas.paste(_c35, (58, 1264), _c35)
except Exception:
    pass
layout["Watch_on_YouTube"] = [58, 1264, 296, 1329]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/36_clickable_Share.png
try:
    _c36 = get_crop(36, 65, 84)
    canvas.paste(_c36, (1285, 1357), _c36)
except Exception:
    pass
layout["Share"] = [1285, 1357, 1350, 1441]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/37_clickable_Photo_image_of_Sociallybuzz_Inc.png
try:
    _c37 = get_crop(37, 66, 66)
    canvas.paste(_c37, (68, 1367), _c37)
except Exception:
    pass
layout["Photo_image_of_Sociallybu"] = [68, 1367, 134, 1433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/38_clickable_Minorities_in_Franchising_Learn_the_7_Ha.png
try:
    _c38 = get_crop(38, 1127, 33)
    canvas.paste(_c38, (144, 1386), _c38)
except Exception:
    pass
layout["Minorities_in_Franchising"] = [144, 1386, 1271, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/39_clickable_Play.png
try:
    _c39 = get_crop(39, 93, 66)
    canvas.paste(_c39, (673, 1702), _c39)
except Exception:
    pass
layout["Play"] = [673, 1702, 766, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_20_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-22/40_clickable_Watch_on_YouTube.png
try:
    _c40 = get_crop(40, 238, 65)
    canvas.paste(_c40, (58, 2040), _c40)
except Exception:
    pass
layout["Watch_on_YouTube"] = [58, 2040, 296, 2105]
