# page_id: page_seatgeek_71f7c21037d54ebf9466fb0a4cb9cb36_01
# screenshot: 2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4.png
# step_index: 1/4
# task: Open SeatGeek. Search for concerts in "New York City". Filter by "pop" genre. What is the second recommendation?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw page background
bg_color = "#fafafa"
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar area (top)
status_h = 84  # ~50px suggested, a bit taller to match screenshot spacing
status_color = "#ececec"
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Status bar bottom divider
divider_color = "#e6e6e6"
draw.line([(0, status_h), (canvas.width, status_h)], fill=divider_color, width=1)

# Header / toolbar area (location + filters)
header_top = status_h
header_bottom = 220  # area containing location text (text will be pasted separately)
draw.rectangle([(0, header_top), (canvas.width, header_bottom)], fill="#ffffff")
# subtle bottom divider under header
draw.line([(24, header_bottom), (canvas.width-24, header_bottom)], fill=divider_color, width=1)

# Large hero card background ("New York Knicks" card) - rounded rectangle with vertical blue gradient
hero_left = 48
hero_top = 360
hero_width = 1344
hero_height = 840
hero_right = hero_left + hero_width
hero_bottom = hero_top + hero_height
radius = 28

# Gradient colors for hero card
top_color = (21, 99, 163)    # deep blue
bottom_color = (46, 142, 199)  # lighter blue

# Draw gradient by horizontal lines within rounded mask region
# Create a mask in-memory approach using draw primitives: draw many rounded rectangles with gradually interpolated fills
# Simpler approach: draw horizontal lines across full width clipped to hero rect, then overlay rounded corners mask using rounded rectangle stroke
for i in range(hero_height):
    t = i / max(hero_height - 1, 1)
    r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
    g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
    b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
    y = hero_top + i
    draw.line([(hero_left, y), (hero_right, y)], fill=(r, g, b))

# Overlay rounded rectangle mask by drawing white outside corners to emulate rounded corners (clear corners)
# Draw a rounded rectangle border in the background color to clip corners visually
# First draw a rounded rectangle in bg color slightly larger around corners then draw the gradient bounded by hero rect minus corners
# Easiest: draw rounded rectangle on top filled with gradient by drawing a rounded rectangle outline in bg to soften corners
# We'll draw a white rounded rectangle outside to cover the squared corners from the gradient
corner_cover = 2
# top-left corner cover
draw.pieslice([(hero_left - corner_cover, hero_top - corner_cover),
               (hero_left + 2*radius - corner_cover, hero_top + 2*radius - corner_cover)],
              start=180, end=270, fill=None, outline=None)
# Instead, draw a white rounded rectangle over the whole area then re-draw inner rounded rectangle with the gradient by using border to produce rounded corners:
# (Since we cannot create complex masks without imports, approximate by drawing rounded rectangle white then draw inner rounded rectangle with gradient using many horizontal lines clipped inside rounded shape)
# Draw a white rounded rectangle as a background to ensure surrounding area is smooth
draw.rounded_rectangle([(hero_left-8, hero_top-8), (hero_right+8, hero_bottom+8)], radius=radius+8, fill="#ffffff")

# Now draw the hero rounded rectangle border (slightly inset) to reveal the gradient inside with rounded corners
# We'll clip by drawing rounded rectangle outline in gradient color: re-draw gradient but only inside an inset rounded rect by skipping few pixels near edges
inset = 0
for i in range(hero_height - inset*2):
    t = i / max(hero_height - 1, 1)
    r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
    g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
    b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
    y = hero_top + inset + i
    # compute horizontal span for rounded corners
    # simple approach: draw full horizontal line but then overlay white rectangles on the four corners to fake rounding
    draw.line([(hero_left+inset, y), (hero_right-inset, y)], fill=(r, g, b))

# Draw final rounded rectangle outline to smooth corners (no fill)
draw.rounded_rectangle([(hero_left, hero_top), (hero_right, hero_bottom)], radius=radius, outline=None, width=0)

# Card inner badge background (a subtle darker strip near bottom of hero) - just background strip (no text)
strip_h = 80
strip_color = (0, 0, 0, 30)  # semi-transparent black not supported directly; approximate with very dark semi
# Use a darker blue rectangle aligned near the bottom of hero to suggest the badge area
strip_top = hero_bottom - 120
draw.rounded_rectangle([(hero_left + 80, strip_top), (hero_right - 80, strip_top + strip_h)], radius=12, fill=(12, 73, 127))

# "Just for you" section background card (rounded white card to hold the thumbnails)
just_section_top = 1280
just_section_left = 24
just_section_right = canvas.width - 24
just_section_bottom = 2000
draw.rounded_rectangle([(just_section_left, just_section_top), (just_section_right, just_section_bottom)],
                       radius=20, fill="#ffffff")
# subtle shadow line under this section
draw.line([(just_section_left+12, just_section_bottom), (just_section_right-12, just_section_bottom)], fill=divider_color, width=1)

# Draw separators between the small thumbnail row and the trending header
thumbs_row_bottom = 1431 + 519  # from detection: top 1431, height 519
# horizontal divider below thumbnails
sep_y1 = thumbs_row_bottom + 24
draw.line([(24, sep_y1), (canvas.width-24, sep_y1)], fill=divider_color, width=1)

# Trending header background area
trending_header_top = sep_y1 + 36
trending_header_bottom = 2040
draw.rectangle([(0, trending_header_top), (canvas.width, trending_header_bottom)], fill="#ffffff")
# divider lines around trending header
draw.line([(24, trending_header_bottom), (canvas.width-24, trending_header_bottom)], fill=divider_color, width=1)
draw.line([(24, trending_header_top), (canvas.width-24, trending_header_top)], fill=divider_color, width=1)

# Trending list rows background (subtle white panels and separators)
# Use positions approximated from detected trending rows:
trend_row1_top = 2183
trend_row_h = 236
for i in range(3):  # draw three row backgrounds (actual content will be pasted on top)
    top = trend_row1_top + i * trend_row_h
    bottom = top + trend_row_h
    # white background for each row (keep full width minus small margins)
    draw.rectangle([(24, top+12), (canvas.width-24, bottom-12)], fill="#ffffff")
    # separators between rows
    draw.line([(36, bottom), (canvas.width-36, bottom)], fill=divider_color, width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (canvas.width, canvas.height)], fill="#ffffff")
draw.line([(0, nav_top), (canvas.width, nav_top)], fill=divider_color, width=1)

# Subtle left and right edge rounded accents to mimic carousel peek areas (don't overlap detected image areas significantly)
edge_peek_w = 28
peek_color_left = (250, 192, 64)  # warm gold peek on right side of hero, but keep subtle
# small rounded stripe on right edge near hero middle
draw.rounded_rectangle([(canvas.width - 24 - edge_peek_w, hero_top + 40), (canvas.width - 24, hero_bottom - 40)],
                       radius=12, fill="#b58500")

# Small decorative horizontal separators lower in the page to structure content (non-intrusive)
for y in (2300, 2470, 2630):
    draw.line([(48, y), (canvas.width - 48, y)], fill="#f1f1f1", width=1)

# Final subtle vignette lines to give depth (very faint)
for i, a in enumerate([1, 2, 3]):
    y = hero_bottom + 12 + i * 8
    draw.line([(48, y), (canvas.width - 48, y)], fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/00_icon_Knicks.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Knicks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/01_icon_BOOK_OF.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (48, 1431), _c1)
except Exception:
    pass
layout["BOOK_OF"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/02_icon_August_Wilson_Theatre.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 2183), _c2)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/03_icon_Yankee_Stadium.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2419), _c3)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/04_icon_S116.png
try:
    _c4 = get_crop(4, 396, 519)
    canvas.paste(_c4, (1044, 1431), _c4)
except Exception:
    pass
layout["S116+"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/05_icon_S94.png
try:
    _c5 = get_crop(5, 462, 519)
    canvas.paste(_c5, (546, 1431), _c5)
except Exception:
    pass
layout["S94+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 99, 152)
    canvas.paste(_c6, (1341, 2464), _c6)
except Exception:
    pass
layout["icon_6"] = [1341, 2464, 1440, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/07_icon_View_all.png
try:
    _c7 = get_crop(7, 98, 149)
    canvas.paste(_c7, (1342, 2228), _c7)
except Exception:
    pass
layout["View_all"] = [1342, 2228, 1440, 2377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/08_icon_New_York_NY.png
try:
    _c8 = get_crop(8, 61, 58)
    canvas.paste(_c8, (243, 5), _c8)
except Exception:
    pass
layout["New_York,_NY"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/09_icon_May.png
try:
    _c9 = get_crop(9, 264, 183)
    canvas.paste(_c9, (1176, 2000), _c9)
except Exception:
    pass
layout["May"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/10_icon_888.png
try:
    _c10 = get_crop(10, 99, 63)
    canvas.paste(_c10, (1214, 1), _c10)
except Exception:
    pass
layout["888"] = [1214, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/11_icon_7_02_my.png
try:
    _c11 = get_crop(11, 55, 57)
    canvas.paste(_c11, (114, 5), _c11)
except Exception:
    pass
layout["7:02_my"] = [114, 5, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/12_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (864, 2792), _c12)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/13_icon_7_02_my.png
try:
    _c13 = get_crop(13, 47, 57)
    canvas.paste(_c13, (185, 5), _c13)
except Exception:
    pass
layout["7:02_my"] = [185, 5, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/14_icon_888.png
try:
    _c14 = get_crop(14, 144, 240)
    canvas.paste(_c14, (1260, 72), _c14)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/15_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (288, 2792), _c15)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 63)
    canvas.paste(_c16, (1320, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [1320, 2, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/17_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (576, 2792), _c17)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 54, 59)
    canvas.paste(_c18, (314, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [314, 5, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 46, 64)
    canvas.paste(_c19, (1154, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [1154, 1, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 99, 119)
    canvas.paste(_c20, (1341, 2698), _c20)
except Exception:
    pass
layout["icon_20"] = [1341, 2698, 1440, 2817]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/21_icon_Browse.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (0, 2792), _c21)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/22_icon_Account.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (1152, 2792), _c22)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/23_icon_Andrew_Schulz.png
try:
    _c23 = get_crop(23, 462, 519)
    canvas.paste(_c23, (546, 1431), _c23)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 116, 127)
    canvas.paste(_c24, (1138, 2484), _c24)
except Exception:
    pass
layout["icon_24"] = [1138, 2484, 1254, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/25_icon_New_York_NY.png
try:
    _c25 = get_crop(25, 390, 86)
    canvas.paste(_c25, (40, 119), _c25)
except Exception:
    pass
layout["New_York,_NY"] = [40, 119, 430, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/26_icon_The.png
try:
    _c26 = get_crop(26, 91, 102)
    canvas.paste(_c26, (36, 1427), _c26)
except Exception:
    pass
layout["The"] = [36, 1427, 127, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/27_text_date.png
try:
    _c27 = get_crop(27, 114, 52)
    canvas.paste(_c27, (137, 208), _c27)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/28_text_Just_for_you.png
try:
    _c28 = get_crop(28, 306, 66)
    canvas.paste(_c28, (38, 1310), _c28)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/29_text_View_all.png
try:
    _c29 = get_crop(29, 264, 183)
    canvas.paste(_c29, (1176, 1248), _c29)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/30_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c30 = get_crop(30, 288, 168)
    canvas.paste(_c30, (576, 2792), _c30)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (408, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_01_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-4/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (906, 1455), _c32)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
