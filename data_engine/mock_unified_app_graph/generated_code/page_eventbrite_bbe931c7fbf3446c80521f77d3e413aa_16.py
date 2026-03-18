# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_16
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18.png
# step_index: 16/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structure for Eventbrite-like mobile UI
# Canvas: 1440x2960 (provided as `canvas`), draw: ImageDraw object, fonts available

# Colors
status_bar_color = (169, 169, 169)      # muted grey for status bar
header_bg = (255, 255, 255)             # white header
page_bg = (250, 250, 252)               # very light off-white page background
divider_color = (226, 226, 230)         # subtle divider
card_border_blue = (57, 88, 255)        # bluish card outline for ticket card
card_bg = (255, 255, 255)               # white card background
shadow_color = (235, 235, 238)          # light shadow line
safe_area_bg = (247, 247, 249)          # bottom safe-area tint

W, H = canvas.size

# Fill overall page background (covers any existing white)
draw.rectangle([(0,0),(W,H)], fill=page_bg)

# Status bar area at top (~0-90)
status_h = 90
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_color)

# Header / toolbar background (below status bar)
header_top = status_h
header_bottom = 220
draw.rectangle([(0,header_top),(W,header_bottom)], fill=header_bg)

# Subtle bottom divider/shadow for header
draw.line([(32, header_bottom), (W-32, header_bottom)], fill=divider_color, width=1)
draw.line([(32, header_bottom+1), (W-32, header_bottom+1)], fill=shadow_color, width=1)

# Left and right safe margins for content
margin_x = 48
content_left = margin_x
content_right = W - margin_x

# Section separators (subtle horizontal rules) across content area where sections would split
# These do not overlap detected UI elements directly; they are generic separators.
separators = [640, 960, 1540, 2160]  # approximate y positions for section breaks
for y in separators:
    draw.line([(content_left+8, y), (content_right-8, y)], fill=divider_color, width=1)

# Large subtle banner / content area background where main body text sits
# Keep generous top offset to avoid header icons/title area (detected icons at ~108 y)
body_top = header_bottom + 20
body_bottom = 2300
draw.rectangle([(content_left-10, body_top), (content_right+10, body_bottom)], fill=page_bg)

# Thin left and right guide bars (very subtle) to frame content columns
guide_color = (245, 245, 247)
draw.rectangle([(0, body_top), (content_left-8, body_bottom)], fill=guide_color)
draw.rectangle([(content_right+8, body_top), (W, body_bottom)], fill=guide_color)

# Ticket selection card (rounded rectangle) above Reserve button area
card_top = 2300
card_bottom = 2632
card_rect = [ (content_left, card_top), (content_right, card_bottom) ]
card_radius = 28
# Outer border
draw.rounded_rectangle(card_rect, radius=card_radius, fill=card_bg, outline=card_border_blue, width=8)
# Inner subtle highlight (a very thin white inset to simulate card depth)
inset = 10
draw.rounded_rectangle([(content_left+inset, card_top+inset), (content_right-inset, card_bottom-inset)],
                       radius=max(1, card_radius-inset), outline=(255,255,255), width=2)

# Divider within card to separate header text area and controls (approx)
inner_div_y = card_top + 120
draw.line([(content_left+24, inner_div_y), (content_right-24, inner_div_y)], fill=divider_color, width=1)

# Slight drop shadow under the card to lift it visually
shadow_y0 = card_bottom
shadow_y1 = card_bottom + 18
for i, alpha_shade in enumerate([235, 228, 220, 210]):
    y = shadow_y0 + i
    shade = tuple(int(c*(alpha_shade/255.0) + 255*(1 - alpha_shade/255.0)) for c in (220,220,220))
    draw.line([(content_left+6, y), (content_right-6, y)], fill=shade, width=1)

# Safe area background at very bottom (to visually separate from reserve button)
safe_top = card_bottom + 140
draw.rectangle([(0, safe_top), (W, H)], fill=safe_area_bg)

# Top-of-content subtle horizontal rule (separator under main title area)
top_content_div = header_bottom + 12
draw.line([(content_left, top_content_div), (content_right, top_content_div)], fill=divider_color, width=1)

# Decorative faint left accent (vertical) for content column
accent_x = content_left + 6
draw.rectangle([(accent_x, body_top+20), (accent_x+6, body_top+360)], fill=(234,234,239))

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/02_icon_LGan_UVV.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["LGan_UVV"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/03_icon_Decrease.png
try:
    _c3 = get_crop(3, 99, 96)
    canvas.paste(_c3, (996, 2444), _c3)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 54)
    canvas.paste(_c4, (316, 8), _c4)
except Exception:
    pass
layout["icon_4"] = [316, 8, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 54)
    canvas.paste(_c5, (250, 7), _c5)
except Exception:
    pass
layout["icon_5"] = [250, 7, 300, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/06_icon_Increase.png
try:
    _c6 = get_crop(6, 96, 96)
    canvas.paste(_c6, (1224, 2444), _c6)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 93, 103)
    canvas.paste(_c7, (1107, 2441), _c7)
except Exception:
    pass
layout["icon_7"] = [1107, 2441, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/08_icon_Reserve_a_spot.png
try:
    _c8 = get_crop(8, 1296, 132)
    canvas.paste(_c8, (72, 2756), _c8)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 53)
    canvas.paste(_c9, (184, 7), _c9)
except Exception:
    pass
layout["icon_9"] = [184, 7, 234, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 44, 62)
    canvas.paste(_c10, (1157, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [1157, 4, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/11_icon_9.13.png
try:
    _c11 = get_crop(11, 51, 57)
    canvas.paste(_c11, (118, 5), _c11)
except Exception:
    pass
layout["9.13"] = [118, 5, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 56)
    canvas.paste(_c12, (1325, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [1325, 5, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 95, 59)
    canvas.paste(_c13, (1215, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1215, 3, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/14_icon_Minorities_Building-.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (36, 108), _c14)
except Exception:
    pass
layout["Minorities_Building-"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 43, 56)
    canvas.paste(_c15, (386, 6), _c15)
except Exception:
    pass
layout["icon_15"] = [386, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/16_icon_Free.png
try:
    _c16 = get_crop(16, 137, 115)
    canvas.paste(_c16, (98, 2567), _c16)
except Exception:
    pass
layout["Free"] = [98, 2567, 235, 2682]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/17_icon_includes_effective_campaigns_with_brands.png
try:
    _c17 = get_crop(17, 99, 96)
    canvas.paste(_c17, (996, 2444), _c17)
except Exception:
    pass
layout["includes_effective_campai"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/18_text_9.13.png
try:
    _c18 = get_crop(18, 91, 43)
    canvas.paste(_c18, (20, 17), _c18)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/19_text_LGan_UVV.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["LGan_UVV"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/20_text_Lu_YGL_allGjj_lU_lapIlaI.png
try:
    _c20 = get_crop(20, 487, 41)
    canvas.paste(_c20, (299, 253), _c20)
except Exception:
    pass
layout["Lu_YGL_allGjj_lU_lapIlaI"] = [299, 253, 786, 294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/21_text_Learn_the_advantages_and_disadvantages_o.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1116, 108), _c21)
except Exception:
    pass
layout["Learn_the_advantages_and_"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/22_text_Learn.png
try:
    _c22 = get_crop(22, 126, 49)
    canvas.paste(_c22, (72, 551), _c22)
except Exception:
    pass
layout["Learn"] = [72, 551, 198, 600]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/23_text_Learn_the_current_state_of.png
try:
    _c23 = get_crop(23, 540, 52)
    canvas.paste(_c23, (72, 675), _c23)
except Exception:
    pass
layout["Learn_the_current_state_o"] = [72, 675, 612, 727]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/24_text_Understanding_the_legal_language_of_fran.png
try:
    _c24 = get_crop(24, 1027, 84)
    canvas.paste(_c24, (69, 787), _c24)
except Exception:
    pass
layout["Understanding_the_legal_l"] = [69, 787, 1096, 871]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/25_text_Speakers.png
try:
    _c25 = get_crop(25, 224, 67)
    canvas.paste(_c25, (40, 1051), _c25)
except Exception:
    pass
layout["Speakers:"] = [40, 1051, 264, 1118]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/26_text_1_Andre.png
try:
    _c26 = get_crop(26, 193, 54)
    canvas.paste(_c26, (46, 1180), _c26)
except Exception:
    pass
layout["1)_Andre"] = [46, 1180, 239, 1234]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/27_text_CEO_of_Sociallybuzz._Founded_in_2009_we_.png
try:
    _c27 = get_crop(27, 997, 69)
    canvas.paste(_c27, (348, 1174), _c27)
except Exception:
    pass
layout["CEO_of_Sociallybuzz._Foun"] = [348, 1174, 1345, 1243]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/28_text_Andre.png
try:
    _c28 = get_crop(28, 133, 50)
    canvas.paste(_c28, (46, 1371), _c28)
except Exception:
    pass
layout["Andre"] = [46, 1371, 179, 1421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/29_text_is_an_acclaimed_entrepreneur_and_innovat.png
try:
    _c29 = get_crop(29, 1037, 64)
    canvas.paste(_c29, (260, 1370), _c29)
except Exception:
    pass
layout["is_an_acclaimed_entrepren"] = [260, 1370, 1297, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/30_text_media_agency_launched_in_2008._His_visio.png
try:
    _c30 = get_crop(30, 1289, 72)
    canvas.paste(_c30, (44, 1490), _c30)
except Exception:
    pass
layout["media_agency_launched_in_"] = [44, 1490, 1333, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/31_text_best_entrepreneurial_firms_in_America._B.png
try:
    _c31 = get_crop(31, 99, 96)
    canvas.paste(_c31, (996, 2444), _c31)
except Exception:
    pass
layout["best_entrepreneurial_firm"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/32_text_underscores_his_success..png
try:
    _c32 = get_crop(32, 533, 52)
    canvas.paste(_c32, (42, 1748), _c32)
except Exception:
    pass
layout["underscores_his_success."] = [42, 1748, 575, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_16_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-18/33_text_Complimentary_Access.png
try:
    _c33 = get_crop(33, 75, 72)
    canvas.paste(_c33, (249, 2588), _c33)
except Exception:
    pass
layout["Complimentary_Access"] = [249, 2588, 324, 2660]
