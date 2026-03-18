# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_08
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10.png
# step_index: 8/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural chrome for the mobile UI
# (uses provided canvas and draw objects)

# Colors
status_bar_color = (191, 191, 191)       # muted grey for status bar
header_cream = (249, 243, 239)           # warm cream header/banner
hero_strip = (236, 229, 224)             # slightly darker band under header
card_bg = (247, 246, 250)                # very light card background (lavender/grey)
card_shadow = (225, 223, 226)            # subtle shadow for cards
divider = (230, 228, 232)                # faint divider lines
pill_bg = (237, 244, 250)                # pale blue pill for category background
footer_bg = (250, 250, 251)              # sticky footer background

W, H = canvas.size

# 1) Status bar (top ~56px)
draw.rectangle([0, 0, W, 56], fill=status_bar_color)

# 2) Main header/banner area (cream background) under status bar
header_top = 56
header_bottom = 420
draw.rectangle([0, header_top, W, header_bottom], fill=header_cream)

# subtle horizontal band (hero image area) to suggest the photo strip under the title
hero_top = header_top + 140
hero_bottom = header_top + 320
draw.rectangle([0, hero_top, W, hero_bottom], fill=hero_strip)

# a faint bottom border under the hero band
draw.line([0, hero_bottom, W, hero_bottom], fill=divider, width=1)

# 3) Organizer / Follow rounded card (do not draw buttons/icons/text inside)
card_left = 36
card_right = W - 36
card_top = 980
card_bottom = 1140
# shadow slightly below
draw.rounded_rectangle(
    [card_left + 4, card_top + 6, card_right + 4, card_bottom + 6],
    radius=28, fill=card_shadow
)
draw.rounded_rectangle(
    [card_left, card_top, card_right, card_bottom],
    radius=28, fill=card_bg
)

# subtle inner divider on the card (to separate avatar area from follow button area visually)
draw.line([card_left + 180, card_top + 20, card_left + 180, card_bottom - 20], fill=(245,244,246), width=1)

# 4) Small separators/dividers across the content area
# under date/title area
draw.line([48, 900, W - 48, 900], fill=divider, width=1)
# under refund/policy section
draw.line([48, 1670, W - 48, 1670], fill=divider, width=1)
# under "About this event" header area
draw.line([48, 1960, W - 48, 1960], fill=divider, width=1)

# 5) Category/tag pill background (behind "Community & Culture" label)
pill_x = 48
pill_y = 2170
pill_w = 300
pill_h = 68
draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h], radius=34, fill=pill_bg)

# 6) Light content area background behind the "About this event" block for subtle separation
about_top = 1840
about_bottom = 2340
draw.rectangle([0, about_top, W, about_bottom], fill=(255, 255, 255))

# 7) Sticky footer background (do not draw ticket button or price text)
footer_top = 2790
draw.rectangle([0, footer_top, W, H], fill=footer_bg)
# Add a faint top border to the footer
draw.line([0, footer_top, W, footer_top], fill=divider, width=1)

# 8) Very bottom center decorative circle (background element behind organizer avatar that peeks above footer)
circle_center_x = W // 2
circle_center_y = footer_top - 80
circle_radius = 120
draw.ellipse(
    [
        circle_center_x - circle_radius,
        circle_center_y - circle_radius,
        circle_center_x + circle_radius,
        circle_center_y + circle_radius,
    ],
    fill=(255, 255, 255),
    outline=(240, 238, 241),
)

# 9) Final subtle vertical padding lines to guide layout (very faint)
draw.line([48, 56, 48, H - 200], fill=(249,249,250), width=1)
draw.line([W - 48, 56, W - 48, H - 200], fill=(249,249,250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/02_icon_Community_Culture.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2205), _c2)
except Exception:
    pass
layout["Community_&_Culture"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/03_icon_4pm_MDT.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["4pm_MDT"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/05_icon_4.42.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["4.42"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 43, 56)
    canvas.paste(_c6, (1327, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [1327, 6, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/07_icon_4.42.png
try:
    _c7 = get_crop(7, 61, 61)
    canvas.paste(_c7, (180, 3), _c7)
except Exception:
    pass
layout["4.42"] = [180, 3, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 95, 56)
    canvas.paste(_c8, (1217, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [1217, 5, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 59, 60)
    canvas.paste(_c9, (311, 4), _c9)
except Exception:
    pass
layout["icon_9"] = [311, 4, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/10_icon_4.42.png
try:
    _c10 = get_crop(10, 59, 62)
    canvas.paste(_c10, (115, 2), _c10)
except Exception:
    pass
layout["4.42"] = [115, 2, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/11_icon_S22.png
try:
    _c11 = get_crop(11, 210, 84)
    canvas.paste(_c11, (1171, 2638), _c11)
except Exception:
    pass
layout["S22"] = [1171, 2638, 1381, 2722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 60)
    canvas.paste(_c12, (247, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 4, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/13_icon_Get_tickets.png
try:
    _c13 = get_crop(13, 267, 263)
    canvas.paste(_c13, (584, 2549), _c13)
except Exception:
    pass
layout["Get_tickets"] = [584, 2549, 851, 2812]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 48, 62)
    canvas.paste(_c14, (383, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [383, 3, 431, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/15_icon_Read_more.png
try:
    _c15 = get_crop(15, 234, 144)
    canvas.paste(_c15, (48, 2205), _c15)
except Exception:
    pass
layout["Read_more"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/16_text_4.42.png
try:
    _c16 = get_crop(16, 89, 41)
    canvas.paste(_c16, (22, 17), _c16)
except Exception:
    pass
layout["4.42"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/17_text_THE_FACES.png
try:
    _c17 = get_crop(17, 436, 88)
    canvas.paste(_c17, (501, 88), _c17)
except Exception:
    pass
layout["THE_FACES"] = [501, 88, 937, 176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/18_text_Worldandliveinharmonizingbalance.png
try:
    _c18 = get_crop(18, 512, 50)
    canvas.paste(_c18, (486, 258), _c18)
except Exception:
    pass
layout["Worldandliveinharmonizing"] = [486, 258, 998, 308]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/19_text_11_2024.png
try:
    _c19 = get_crop(19, 136, 43)
    canvas.paste(_c19, (498, 311), _c19)
except Exception:
    pass
layout["11,_2024"] = [498, 311, 634, 354]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/20_text_Saturday.png
try:
    _c20 = get_crop(20, 252, 77)
    canvas.paste(_c20, (38, 758), _c20)
except Exception:
    pass
layout["Saturday;"] = [38, 758, 290, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/21_text_11.png
try:
    _c21 = get_crop(21, 64, 50)
    canvas.paste(_c21, (407, 770), _c21)
except Exception:
    pass
layout["11"] = [407, 770, 471, 820]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/22_text_12_00_PM.png
try:
    _c22 = get_crop(22, 236, 56)
    canvas.paste(_c22, (514, 766), _c22)
except Exception:
    pass
layout["12:00_PM"] = [514, 766, 750, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/23_text_The_Faces_of_Feminine.png
try:
    _c23 = get_crop(23, 329, 144)
    canvas.paste(_c23, (288, 1068), _c23)
except Exception:
    pass
layout["The_Faces_of_Feminine"] = [288, 1068, 617, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/24_text_Medicine_Muse.png
try:
    _c24 = get_crop(24, 329, 144)
    canvas.paste(_c24, (288, 1068), _c24)
except Exception:
    pass
layout["Medicine_Muse"] = [288, 1068, 617, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/25_text_Online_event.png
try:
    _c25 = get_crop(25, 275, 56)
    canvas.paste(_c25, (138, 1341), _c25)
except Exception:
    pass
layout["Online_event"] = [138, 1341, 413, 1397]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/26_text_4hrs.png
try:
    _c26 = get_crop(26, 112, 50)
    canvas.paste(_c26, (141, 1452), _c26)
except Exception:
    pass
layout["4hrs"] = [141, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/27_text_Refund_policy.png
try:
    _c27 = get_crop(27, 299, 63)
    canvas.paste(_c27, (138, 1558), _c27)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/28_text_No_refunds.png
try:
    _c28 = get_crop(28, 214, 49)
    canvas.paste(_c28, (139, 1649), _c28)
except Exception:
    pass
layout["No_refunds"] = [139, 1649, 353, 1698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/29_text_About_this_event.png
try:
    _c29 = get_crop(29, 454, 61)
    canvas.paste(_c29, (45, 1858), _c29)
except Exception:
    pass
layout["About_this_event"] = [45, 1858, 499, 1919]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/30_text_Join_us_for_a_celebration_of_the_diverse.png
try:
    _c30 = get_crop(30, 234, 144)
    canvas.paste(_c30, (48, 2205), _c30)
except Exception:
    pass
layout["Join_us_for_a_celebration"] = [48, 2205, 282, 2349]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/31_text_S55.png
try:
    _c31 = get_crop(31, 99, 55)
    canvas.paste(_c31, (90, 2814), _c31)
except Exception:
    pass
layout["S55"] = [90, 2814, 189, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/32_clickable_Organizer_profile_picture.png
try:
    _c32 = get_crop(32, 144, 144)
    canvas.paste(_c32, (96, 1067), _c32)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_08_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-10/33_clickable_Location.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1295), _c33)
except Exception:
    pass
layout["Location"] = [48, 1295, 1392, 1439]
