# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_19
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21.png
# step_index: 19/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas & draw are provided (PIL Image and ImageDraw)
W, H = canvas.size

# Colors
status_bar_color = (208, 208, 208)    # light grey status bar
hero_top_color = (30, 6, 40)          # deep purple
hero_bottom_color = (6, 0, 0)         # near black for banner
content_divider = (236, 236, 239)     # very light divider
card_bg = (246, 245, 250)             # light card background
panel_bg = (242, 241, 244)            # bottom panel background
shadow_line = (220, 218, 225)         # subtle shadow color

# 1) Status bar area at top (~80px)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# 2) Hero/banner area gradient (below status bar)
hero_y0 = status_h
hero_y1 = 480
hero_h = hero_y1 - hero_y0

# Draw vertical gradient for hero/header image background
for i in range(hero_h):
    t = i / max(hero_h - 1, 1)
    r = int(hero_top_color[0] * (1 - t) + hero_bottom_color[0] * t)
    g = int(hero_top_color[1] * (1 - t) + hero_bottom_color[1] * t)
    b = int(hero_top_color[2] * (1 - t) + hero_bottom_color[2] * t)
    draw.line([(0, hero_y0 + i), (W, hero_y0 + i)], fill=(r, g, b))

# Slight vignette: darker left & right vertical overlays (subtle)
vignette_w = 220
for i in range(vignette_w):
    alpha = int(120 * (1 - i / vignette_w))  # pseudo-alpha effect via darker lines
    overlay_color = (0, 0, 0)
    draw.line([(i, hero_y0), (i, hero_y1)], fill=(
        max(0, hero_bottom_color[0] - alpha // 6),
        max(0, hero_bottom_color[1] - alpha // 6),
        max(0, hero_bottom_color[2] - alpha // 6)
    ))
    j = W - 1 - i
    draw.line([(j, hero_y0), (j, hero_y1)], fill=(
        max(0, hero_bottom_color[0] - alpha // 6),
        max(0, hero_bottom_color[1] - alpha // 6),
        max(0, hero_bottom_color[2] - alpha // 6)
    ))

# 3) Thin divider below hero
draw.line([(32, hero_y1), (W - 32, hero_y1)], fill=content_divider, width=2)

# 4) Main content background is white by default; add subtle large content area band
content_y0 = hero_y1 + 20
# No fill needed (canvas is white), but add a soft horizontal rule under header area
draw.line([(48, content_y0 + 420), (W - 48, content_y0 + 420)], fill=content_divider, width=1)

# 5) Organizer / profile card background (rounded rectangle)
card_x1 = 48
card_x2 = W - 48
card_y1 = 1210
card_y2 = 1360
card_radius = 28
# card background
draw.rounded_rectangle([(card_x1, card_y1), (card_x2, card_y2)], radius=card_radius, fill=card_bg, outline=None)

# subtle top shadow line for the card
draw.line([(card_x1 + 6, card_y1 + 2), (card_x2 - 6, card_y1 + 2)], fill=shadow_line)

# subtle bottom separator under card
draw.line([(card_x1 + 12, card_y2 + 10), (card_x2 - 12, card_y2 + 10)], fill=content_divider, width=1)

# 6) Small separators between info rows (do not draw icons or text)
sep_x = 48
sep_x2 = W - 48
# Separator under event info block
sep_y1 = 1680
draw.line([(sep_x, sep_y1), (sep_x2, sep_y1)], fill=content_divider, width=1)

# Separator under "About this event" area
sep_y2 = 2440
draw.line([(sep_x, sep_y2), (sep_x2, sep_y2)], fill=content_divider, width=1)

# 7) Location / map area background: subtle off-white band behind location section
loc_band_y0 = 2580
loc_band_y1 = 2720
draw.rectangle([(0, loc_band_y0), (W, loc_band_y1)], fill=(255, 255, 255))

# 8) Bottom ticket panel background (leave button area for detected element to be pasted)
bottom_panel_y = 2760
draw.rectangle([(0, bottom_panel_y), (W, H)], fill=panel_bg)

# subtle top border for bottom panel
draw.line([(24, bottom_panel_y), (W - 24, bottom_panel_y)], fill=content_divider, width=2)

# 9) Left price area box on bottom panel (simple outline / subtle highlight) - avoid drawing on exact text positions
price_box_x1 = 48
price_box_x2 = 700  # keep well left of the actual 'Get tickets' button area (which starts at ~822)
price_box_y1 = bottom_panel_y + 16
price_box_y2 = H - 16
draw.rectangle([(price_box_x1, price_box_y1), (price_box_x2, price_box_y2)], outline=shadow_line, width=1, fill=None)

# 10) Decorative thin rule above large content sections for visual structure
draw.line([(48, 920), (W - 48, 920)], fill=content_divider, width=1)

# 11) Add a faint vertical guide line left to separate meta icons area from text blocks
meta_x = 96
draw.line([(meta_x, hero_y1 + 60), (meta_x, sep_y1)], fill=(250, 250, 250), width=1)

# NOTE: Intentionally do NOT draw any text, icons, or button artwork that will be pasted later.
# The shapes above provide only background, cards, banners, separators and panels.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/02_icon_BIH.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["BIH"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/03_icon_DANGE.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["DANGE"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/04_icon_Early_bird_discount.png
try:
    _c4 = get_crop(4, 449, 144)
    canvas.paste(_c4, (48, 724), _c4)
except Exception:
    pass
layout["Early_bird_discount"] = [48, 724, 497, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/05_icon_DANGE.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["DANGE"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 64)
    canvas.paste(_c6, (1156, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1156, 3, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/07_icon_Music.png
try:
    _c7 = get_crop(7, 203, 101)
    canvas.paste(_c7, (41, 2166), _c7)
except Exception:
    pass
layout["Music"] = [41, 2166, 244, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 61)
    canvas.paste(_c8, (1327, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1327, 3, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/09_icon_NIGHTCLUB.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1116, 108), _c9)
except Exception:
    pass
layout["NIGHTCLUB"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 96, 63)
    canvas.paste(_c10, (1215, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1215, 2, 1311, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/11_icon_BIH.png
try:
    _c11 = get_crop(11, 60, 68)
    canvas.paste(_c11, (181, 1), _c11)
except Exception:
    pass
layout["BIH"] = [181, 1, 241, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/12_icon_Bollywood_night_at_the_the_1_night_club_.png
try:
    _c12 = get_crop(12, 234, 144)
    canvas.paste(_c12, (48, 2372), _c12)
except Exception:
    pass
layout["Bollywood_night_at_the_th"] = [48, 2372, 282, 2516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/13_icon_7.48.png
try:
    _c13 = get_crop(13, 60, 69)
    canvas.paste(_c13, (115, 0), _c13)
except Exception:
    pass
layout["7.48"] = [115, 0, 175, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/14_icon_BIH.png
try:
    _c14 = get_crop(14, 57, 69)
    canvas.paste(_c14, (246, 1), _c14)
except Exception:
    pass
layout["BIH"] = [246, 1, 303, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/15_icon_BIH.png
try:
    _c15 = get_crop(15, 69, 70)
    canvas.paste(_c15, (307, 0), _c15)
except Exception:
    pass
layout["BIH"] = [307, 0, 376, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/16_icon_BIH.png
try:
    _c16 = get_crop(16, 143, 107)
    canvas.paste(_c16, (170, 99), _c16)
except Exception:
    pass
layout["BIH"] = [170, 99, 313, 206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/17_icon_Show_map.png
try:
    _c17 = get_crop(17, 226, 144)
    canvas.paste(_c17, (1166, 2590), _c17)
except Exception:
    pass
layout["Show_map"] = [1166, 2590, 1392, 2734]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/18_text_7.48.png
try:
    _c18 = get_crop(18, 89, 41)
    canvas.paste(_c18, (22, 17), _c18)
except Exception:
    pass
layout["7.48"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/19_text_Friday_May_3_._10.00_PM.png
try:
    _c19 = get_crop(19, 449, 144)
    canvas.paste(_c19, (48, 724), _c19)
except Exception:
    pass
layout["Friday;_May_3_._10.00_PM"] = [48, 724, 497, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/20_text_Bollywood_Takeover_One_Last_Dance.png
try:
    _c20 = get_crop(20, 449, 144)
    canvas.paste(_c20, (48, 724), _c20)
except Exception:
    pass
layout["Bollywood_Takeover:_One_L"] = [48, 724, 497, 868]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/21_text_Temple_Nightclub_SF.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (111, 1250), _c21)
except Exception:
    pass
layout["Temple_Nightclub_SF"] = [111, 1250, 255, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/22_text_BIH.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (111, 1250), _c22)
except Exception:
    pass
layout["BIH"] = [111, 1250, 255, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/23_text_2.Ik_Followers.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (111, 1250), _c23)
except Exception:
    pass
layout["2.Ik_Followers"] = [111, 1250, 255, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/24_text_Temple_Nightclub_San_Francisco.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1517), _c24)
except Exception:
    pass
layout["Temple_Nightclub_San_Fran"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/25_text_4hrs.png
try:
    _c25 = get_crop(25, 112, 50)
    canvas.paste(_c25, (141, 1674), _c25)
except Exception:
    pass
layout["4hrs"] = [141, 1674, 253, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/26_text_Refund_policy.png
try:
    _c26 = get_crop(26, 299, 63)
    canvas.paste(_c26, (138, 1780), _c26)
except Exception:
    pass
layout["Refund_policy"] = [138, 1780, 437, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/27_text_No_refunds.png
try:
    _c27 = get_crop(27, 214, 49)
    canvas.paste(_c27, (139, 1871), _c27)
except Exception:
    pass
layout["No_refunds"] = [139, 1871, 353, 1920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/28_text_About_this_event.png
try:
    _c28 = get_crop(28, 452, 57)
    canvas.paste(_c28, (46, 2081), _c28)
except Exception:
    pass
layout["About_this_event"] = [46, 2081, 498, 2138]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/29_text_Location.png
try:
    _c29 = get_crop(29, 246, 61)
    canvas.paste(_c29, (41, 2635), _c29)
except Exception:
    pass
layout["Location"] = [41, 2635, 287, 2696]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_19_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-21/30_text_S20_-_100.png
try:
    _c30 = get_crop(30, 255, 61)
    canvas.paste(_c30, (89, 2811), _c30)
except Exception:
    pass
layout["S20_-_$100"] = [89, 2811, 344, 2872]
