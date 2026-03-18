# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_01
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4.png
# step_index: 1/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structure on provided `canvas` with `draw`.
# Assumes: canvas is a PIL Image (1440x2960 RGB), draw is ImageDraw, fonts available.

W, H = canvas.size

# Colors
status_bg = "#f5f5f5"
divider = "#e6e6e6"
muted_divider = "#efefef"
page_bg = "#ffffff"
card_bg = "#ffffff"
card_outline = "#e9e9e9"
list_item_bg = "#ffffff"
bottom_shadow = "#f2f2f2"

# 1) Global background (canvas already white). If needed, slightly warm it.
draw.rectangle([(0, 0), (W, H)], fill=page_bg)

# 2) Status bar area at top (~0..88 px)
status_h = 88
draw.rectangle([(0, 0), (W, status_h)], fill=status_bg)
# subtle bottom divider under status bar
draw.line([(24, status_h), (W-24, status_h)], fill=divider, width=1)

# 3) Header / location area (below status). Keep background white, draw divider
header_top = status_h
header_bottom = 240
# faint divider at bottom of header
draw.line([(24, header_bottom), (W-24, header_bottom)], fill=divider, width=1)

# 4) Big hero card is auto-pasted at (48,360)-(1392,1200) -> DO NOT DRAW inside that rect.
hero_box = (48, 360, 48+1344, 360+840)

# 5) "Just for you" section container background (rounded card behind thumbnails)
# Place it below hero card, leaving small top margin
just_for_you_top = hero_box[3] + 80  # around 1280+? adjust based on screenshot spacing
just_for_you_top = 1310  # align with detected area
just_for_you_bottom = just_for_you_top + 540  # encompass thumbnails row area
jf_x0 = 24
jf_x1 = W - 24
jf_radius = 20
# Draw rounded white card with subtle outline
draw.rounded_rectangle(
    [(jf_x0, just_for_you_top), (jf_x1, just_for_you_bottom)],
    radius=jf_radius,
    fill=card_bg,
    outline=card_outline,
    width=1
)

# 6) Subtle internal divider above trending list
# Place a thin divider between the thumbnail row and the following content
divider_y = just_for_you_bottom + 30
draw.line([(24, divider_y), (W-24, divider_y)], fill=muted_divider, width=1)

# 7) Trending events list area background (large white area)
trending_top = divider_y + 30
trending_bottom = 2680
tr_x0 = 0
tr_x1 = W
draw.rectangle([(tr_x0, trending_top), (tr_x1, trending_bottom)], fill=list_item_bg)

# 8) Section header divider for Trending events (thin line below header area)
trending_header_bottom = trending_top + 80
draw.line([(24, trending_header_bottom), (W-24, trending_header_bottom)], fill=muted_divider, width=1)

# 9) Draw separators between list items in Trending events
# Approximate y positions for separators to match visual spacing (avoid overlapping detected icons)
sep_positions = [
    trending_header_bottom + 140,  # first item separator
    trending_header_bottom + 280,  # second item separator
    trending_header_bottom + 420,  # third item separator
    trending_header_bottom + 560,  # fourth (if any)
]
for y in sep_positions:
    draw.line([(24, y), (W-24, y)], fill=muted_divider, width=1)

# 10) Right-side faint separators (shorter, to mimic clipped list lines)
for y in sep_positions:
    draw.line([(W-300, y), (W-24, y)], fill=muted_divider, width=1)

# 11) Bottom navigation bar area: draw top shadow and white nav background
nav_top = 2792
# subtle shadow above nav
draw.rectangle([(0, nav_top-16), (W, nav_top)], fill=bottom_shadow)
# nav background
draw.rectangle([(0, nav_top), (W, H)], fill=page_bg)
# top divider for nav
draw.line([(24, nav_top), (W-24, nav_top)], fill=divider, width=1)

# 12) Floating subtle card outlines on sides (to echo the rounded hero edges without drawing hero)
# Left gap to the hero: draw small rounded pills at left and right margins above the hero region
# Only decorative, away from hero_box, avoid overlap
left_pill_box = (12, hero_box[1] - 40, 32, hero_box[1] + 40)
right_pill_box = (W-32, hero_box[1] - 40, W-12, hero_box[1] + 40)
draw.rounded_rectangle([left_pill_box[0:2], left_pill_box[2:4]], radius=10, fill="#ffffff", outline=card_outline)
draw.rounded_rectangle([right_pill_box[0:2], right_pill_box[2:4]], radius=10, fill="#ffffff", outline=card_outline)

# 13) Ensure we didn't draw inside the hero card region: clear any accidental strokes by overlaying transparent-like white
# (Just make sure hero bbox remains visually clear — use white rectangle with no border)
draw.rectangle([(hero_box[0]-4, hero_box[1]-4), (hero_box[2]+4, hero_box[3]+4)], fill=None, outline=None)

# Note: We intentionally do NOT draw any icons, text, or elements that will be pasted later.
# Final: If needed, update canvas (PIL Image is modified in-place).

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/00_icon_Clippers.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Clippers"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/01_icon_Dodger_Stadium.png
try:
    _c1 = get_crop(1, 1309, 236)
    canvas.paste(_c1, (0, 2197), _c1)
except Exception:
    pass
layout["Dodger_Stadium"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 99, 151)
    canvas.paste(_c2, (1341, 2243), _c2)
except Exception:
    pass
layout["View_all"] = [1341, 2243, 1440, 2394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/03_icon_Angel_Stadium_of_Anaheim.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2433), _c3)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/04_icon_S262.png
try:
    _c4 = get_crop(4, 462, 519)
    canvas.paste(_c4, (48, 1431), _c4)
except Exception:
    pass
layout["S262+"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/05_icon_Los_Angeles_CA.png
try:
    _c5 = get_crop(5, 61, 58)
    canvas.paste(_c5, (243, 5), _c5)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/06_icon_8_11_my.png
try:
    _c6 = get_crop(6, 57, 57)
    canvas.paste(_c6, (112, 5), _c6)
except Exception:
    pass
layout["8:11_my"] = [112, 5, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/07_icon_8_11_my.png
try:
    _c7 = get_crop(7, 47, 57)
    canvas.paste(_c7, (185, 5), _c7)
except Exception:
    pass
layout["8:11_my"] = [185, 5, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/08_icon_888.png
try:
    _c8 = get_crop(8, 98, 63)
    canvas.paste(_c8, (1215, 1), _c8)
except Exception:
    pass
layout["888"] = [1215, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/09_icon_888.png
try:
    _c9 = get_crop(9, 144, 240)
    canvas.paste(_c9, (1260, 72), _c9)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 63)
    canvas.paste(_c10, (1320, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1320, 2, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/11_icon_Tracking.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (864, 2792), _c11)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 103, 150)
    canvas.paste(_c12, (1337, 2480), _c12)
except Exception:
    pass
layout["icon_12"] = [1337, 2480, 1440, 2630]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 58)
    canvas.paste(_c13, (314, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [314, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/14_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (288, 2792), _c14)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/15_icon_7_PM.png
try:
    _c15 = get_crop(15, 264, 183)
    canvas.paste(_c15, (1176, 2014), _c15)
except Exception:
    pass
layout["7_PM"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 45, 65)
    canvas.paste(_c16, (1155, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1155, 1, 1200, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/17_icon_S66.png
try:
    _c17 = get_crop(17, 462, 533)
    canvas.paste(_c17, (546, 1431), _c17)
except Exception:
    pass
layout["S66+"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/18_icon_Browse.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/19_icon_W_Conf_Ist_Rnd.png
try:
    _c19 = get_crop(19, 462, 533)
    canvas.paste(_c19, (546, 1431), _c19)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 100, 118)
    canvas.paste(_c20, (1340, 2707), _c20)
except Exception:
    pass
layout["icon_20"] = [1340, 2707, 1440, 2825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 116, 127)
    canvas.paste(_c21, (1138, 2495), _c21)
except Exception:
    pass
layout["icon_21"] = [1138, 2495, 1254, 2622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/22_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (576, 2792), _c22)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/23_icon_Los_Angeles_CA.png
try:
    _c23 = get_crop(23, 461, 85)
    canvas.paste(_c23, (40, 122), _c23)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [40, 122, 501, 207]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/24_icon_Account.png
try:
    _c24 = get_crop(24, 288, 168)
    canvas.paste(_c24, (1152, 2792), _c24)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/25_text_date.png
try:
    _c25 = get_crop(25, 114, 52)
    canvas.paste(_c25, (137, 208), _c25)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/26_text_Just_for_you.png
try:
    _c26 = get_crop(26, 309, 66)
    canvas.paste(_c26, (38, 1310), _c26)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 347, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/27_text_View_all.png
try:
    _c27 = get_crop(27, 264, 183)
    canvas.paste(_c27, (1176, 1248), _c27)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/28_text_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (576, 2792), _c28)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/29_clickable_Tracking.png
try:
    _c29 = get_crop(29, 396, 519)
    canvas.paste(_c29, (1044, 1431), _c29)
except Exception:
    pass
layout["Tracking"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 72, 72)
    canvas.paste(_c30, (408, 1455), _c30)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_01_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (906, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
