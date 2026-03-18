# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_08
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10.png
# step_index: 8/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the mobile UI page.
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors (approximate to screenshot)
bg_color = (250, 250, 252)         # very light background
status_bar_color = (190, 190, 190) # grey status bar
header_bg = (255, 255, 255)        # white header/search area
accent_blue = (43, 89, 255)        # underline/search accent
card_bg = (255, 255, 255)          # white card background
card_border = (235, 235, 240)      # subtle card border
thumb_bg = (245, 245, 246)         # thumbnail placeholder background (behind images)
divider = (226, 227, 231)          # section separators
bottom_bar_bg = (255, 255, 255)    # bottom nav background
shadow_line = (240, 240, 243)      # light shadow line for cards

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (top ~64px)
status_h = 64
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / Search area (white area beneath status bar)
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Blue underline for the search field (thin accent line under search)
underline_y = header_bottom - 8
underline_height = 6
draw.rectangle([(48, underline_y), (w - 48, underline_y + underline_height)], fill=accent_blue)

# Thin divider below header/search area
draw.line([(24, header_bottom + 8), (w - 24, header_bottom + 8)], fill=divider, width=1)

# Content list area - draw card backgrounds for each list item
# We'll create a vertical sequence of rounded white cards with subtle borders/shadows.
card_x0 = 48
card_x1 = w - 48
card_width = card_x1 - card_x0
card_height = 220
card_gap = 60
first_card_y = header_bottom + 40

# Draw six cards (matching the number of visible list items in screenshot)
for i in range(6):
    top = first_card_y + i * (card_height + card_gap)
    bottom = top + card_height
    # Slight shadow line at top edge to lift card
    draw.line([(card_x0, top), (card_x1, top)], fill=shadow_line, width=1)
    # Rounded rectangle for card background
    try:
        draw.rounded_rectangle([(card_x0, top), (card_x1, bottom)],
                               radius=12, fill=card_bg, outline=card_border, width=1)
    except AttributeError:
        # Fallback if rounded_rectangle not available: draw normal rectangle with border
        draw.rectangle([(card_x0, top), (card_x1, bottom)], fill=card_bg, outline=card_border)

    # Thumbnail background block on left side of card (will be covered by pasted images)
    thumb_margin = 12
    thumb_w = 180
    thumb_h = card_height - 2 * thumb_margin
    thumb_x0 = card_x0 + thumb_margin
    thumb_y0 = top + thumb_margin
    thumb_x1 = thumb_x0 + thumb_w
    thumb_y1 = thumb_y0 + thumb_h
    # rounded thumb area
    try:
        draw.rounded_rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)],
                               radius=8, fill=thumb_bg, outline=(235,235,235))
    except AttributeError:
        draw.rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], fill=thumb_bg, outline=(235,235,235))

    # Vertical separator line between thumbnail and text area
    sep_x = thumb_x1 + 18
    draw.line([(sep_x, top + 18), (sep_x, bottom - 18)], fill=(245,245,246), width=1)

    # Subtle bottom divider for list continuity
    between_y = bottom + int(card_gap / 2)
    draw.line([(card_x0 + 8, between_y), (card_x1 - 8, between_y)], fill=divider, width=1)

# Footer / bottom navigation bar background
bottom_bar_top = 2800
draw.rectangle([(0, bottom_bar_top), (w, h)], fill=bottom_bar_bg)
# top separator line for bottom nav
draw.line([(0, bottom_bar_top), (w, bottom_bar_top)], fill=divider, width=1)

# Subtle horizontal rule near top of content to separate "Events" label region from list
events_rule_y = header_bottom + 24
draw.line([(24, events_rule_y), (w - 24, events_rule_y)], fill=divider, width=1)

# Small left inset rule under the "Events" title area (visual alignment)
draw.line([(48, events_rule_y + 36), (300, events_rule_y + 36)], fill=accent_blue, width=4)

# Final subtle edge shadows at left and right content edges for depth
draw.line([(card_x0 - 1, header_bottom), (card_x0 - 1, bottom)], fill=shadow_line, width=1)
draw.line([(card_x1 + 1, header_bottom), (card_x1 + 1, bottom)], fill=shadow_line, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/00_icon_GRAPHYGlb.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 390), _c0)
except Exception:
    pass
layout["GRAPHYGlb"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/01_icon_Photographyl.png
try:
    _c1 = get_crop(1, 1344, 191)
    canvas.paste(_c1, (48, 72), _c1)
except Exception:
    pass
layout["Photographyl"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/02_icon_Mon_Aug_12_-_Fri.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2370), _c2)
except Exception:
    pass
layout["Mon,_Aug_12_-_Fri,"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 52, 60)
    canvas.paste(_c3, (315, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [315, 3, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/04_icon_4.50.png
try:
    _c4 = get_crop(4, 51, 63)
    canvas.paste(_c4, (185, 2), _c4)
except Exception:
    pass
layout["4.50"] = [185, 2, 236, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 42, 57)
    canvas.paste(_c5, (254, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [254, 5, 296, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/06_icon_1I_00AM_PDT.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1974), _c6)
except Exception:
    pass
layout["1I:00AM_PDT"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/07_icon_4.50.png
try:
    _c7 = get_crop(7, 59, 63)
    canvas.paste(_c7, (113, 2), _c7)
except Exception:
    pass
layout["4.50"] = [113, 2, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 41, 66)
    canvas.paste(_c8, (1158, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1158, 1, 1199, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/09_icon_anoli.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 1182), _c9)
except Exception:
    pass
layout["@anoli"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/10_icon_Cai.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1974), _c10)
except Exception:
    pass
layout["Cai"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/11_icon_Cancel.png
try:
    _c11 = get_crop(11, 78, 64)
    canvas.paste(_c11, (1218, 1), _c11)
except Exception:
    pass
layout["Cancel"] = [1218, 1, 1296, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/12_icon_4.50.png
try:
    _c12 = get_crop(12, 114, 102)
    canvas.paste(_c12, (59, 119), _c12)
except Exception:
    pass
layout["4.50"] = [59, 119, 173, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/13_icon_Cancel.png
try:
    _c13 = get_crop(13, 149, 144)
    canvas.paste(_c13, (1243, 97), _c13)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/14_icon_8_3834_creator_followers.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (288, 2804), _c14)
except Exception:
    pass
layout["8_3834_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 50, 63)
    canvas.paste(_c15, (1320, 1), _c15)
except Exception:
    pass
layout["Cancel"] = [1320, 1, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/16_icon_Tickets.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (864, 2804), _c16)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/17_icon_Product_Photography_Workshop.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 786), _c17)
except Exception:
    pass
layout["Product_Photography_Works"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1099, 96), _c18)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/19_icon_Student_Photography_Workshop.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1182), _c19)
except Exception:
    pass
layout["Student_Photography_Works"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/20_icon_8_3834_creator_followers.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["8_3834_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/21_icon_Venice_Beach.png
try:
    _c21 = get_crop(21, 224, 53)
    canvas.paste(_c21, (391, 2210), _c21)
except Exception:
    pass
layout["Venice_Beach"] = [391, 2210, 615, 2263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/22_icon_Home.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/23_icon_IO_00_AM_PDT.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1182), _c23)
except Exception:
    pass
layout["IO:00_AM_PDT"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/24_icon_Baseball_Photography_Workshop_with_TCP.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 390), _c24)
except Exception:
    pass
layout["Baseball_Photography_Work"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/25_icon_EDiTc.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1578), _c25)
except Exception:
    pass
layout["EDiTc"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/26_icon_Cancel.png
try:
    _c26 = get_crop(26, 42, 62)
    canvas.paste(_c26, (1271, 2), _c26)
except Exception:
    pass
layout["Cancel"] = [1271, 2, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/28_icon_8_1101_creator_followers.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 390), _c28)
except Exception:
    pass
layout["8_1101_creator_followers"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/29_icon_Product_Photography_Workshop.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 786), _c29)
except Exception:
    pass
layout["Product_Photography_Works"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/30_text_4.50.png
try:
    _c30 = get_crop(30, 89, 43)
    canvas.paste(_c30, (22, 17), _c30)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/31_text_Events.png
try:
    _c31 = get_crop(31, 186, 56)
    canvas.paste(_c31, (46, 301), _c31)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/32_text_503_Brooks_Ave.png
try:
    _c32 = get_crop(32, 256, 38)
    canvas.paste(_c32, (392, 1426), _c32)
except Exception:
    pass
layout["503_Brooks_Ave"] = [392, 1426, 648, 1464]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/33_text_Sat_May_25.png
try:
    _c33 = get_crop(33, 209, 49)
    canvas.paste(_c33, (389, 1633), _c33)
except Exception:
    pass
layout["Sat,_May_25"] = [389, 1633, 598, 1682]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/34_text_1I_O0AM_PDT.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1578), _c34)
except Exception:
    pass
layout["1I:O0AM_PDT"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/35_text_Fashion_Editorial_Photography_with_Canon.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1578), _c35)
except Exception:
    pass
layout["Fashion_Editorial_Photogr"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/36_text_Samy_s_Camera_Pasadena.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1578), _c36)
except Exception:
    pass
layout["Samy's_Camera_Pasadena"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/37_text_8_1037_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1578), _c37)
except Exception:
    pass
layout["8_1037_creator_followers"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/38_text_Sat_Oct_12.png
try:
    _c38 = get_crop(38, 198, 45)
    canvas.paste(_c38, (390, 2030), _c38)
except Exception:
    pass
layout["Sat,_Oct_12"] = [390, 2030, 588, 2075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/39_text_1I_00AM_PDT.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 1974), _c39)
except Exception:
    pass
layout["1I:00AM_PDT"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/40_text_8_1172_creator_followers.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 1974), _c40)
except Exception:
    pass
layout["8_1172_creator_followers"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/41_text_Mon_Aug_12_-_Fri.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 2370), _c41)
except Exception:
    pass
layout["Mon,_Aug_12_-_Fri,"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/42_text_16.png
try:
    _c42 = get_crop(42, 54, 39)
    canvas.paste(_c42, (760, 2459), _c42)
except Exception:
    pass
layout["16"] = [760, 2459, 814, 2498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/43_text_IO_00_AM_PDT.png
try:
    _c43 = get_crop(43, 239, 39)
    canvas.paste(_c43, (834, 2459), _c43)
except Exception:
    pass
layout["IO:00_AM_PDT"] = [834, 2459, 1073, 2498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/44_text_Photography_Make_Your_Own_Camera.png
try:
    _c44 = get_crop(44, 1344, 396)
    canvas.paste(_c44, (48, 2370), _c44)
except Exception:
    pass
layout["Photography:_Make_Your_Ow"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_08_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-10/45_text_8_3834_creator_followers.png
try:
    _c45 = get_crop(45, 1344, 396)
    canvas.paste(_c45, (48, 2370), _c45)
except Exception:
    pass
layout["8_3834_creator_followers"] = [48, 2370, 1392, 2766]
