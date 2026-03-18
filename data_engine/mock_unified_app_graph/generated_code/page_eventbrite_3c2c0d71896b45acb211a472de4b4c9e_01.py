# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_01
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3.png
# step_index: 1/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural scaffolding for the Eventbrite-like UI
# Uses provided variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Page background (soft off-white / very light lavender tint)
bg_color = "#FBF9FF"
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar (top area)
status_h = 120
status_color = "#E6E6E6"  # subtle gray bar
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)
# thin divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#D9D9DB", width=2)

# Header / search area background (behind search box)
# The detected search box is at (195,93) size (1179x144) so we draw a soft header band behind it.
header_band_top = status_h
header_band_bottom = 320
draw.rectangle([(0, header_band_top), (1440, header_band_bottom)], fill=bg_color)

# Search box background (rounded) — draw only the box background and border, not any icons/text
search_left, search_top = 195, 93
search_w, search_h = 1179, 144
search_radius = 72
search_bg = "#FFFFFF"
search_border = "#EDE3F3"  # very light purple/gray border
draw.rounded_rectangle(
    [(search_left, search_top), (search_left + search_w, search_top + search_h)],
    radius=search_radius, fill=search_bg, outline=search_border, width=4
)

# Divider below header area
draw.line([(48, header_band_bottom), (1392, header_band_bottom)], fill="#EEE9F1", width=2)

# Card structure for list items
card_x = 48
card_w = 1344
card_h = 396
card_radius = 20
card_fill = "#FFFFFF"
card_outline = "#F0EAF4"  # very subtle violet outline

# Locations of card top Ys (from detected elements)
card_tops = [490, 886, 1282, 1678, 2074]

for y in card_tops:
    # subtle drop shadow (very light)
    shadow_offset = 8
    shadow_color = "#F5F3F6"
    draw.rounded_rectangle(
        [(card_x + 6, y + shadow_offset, card_x + card_w + 6, y + card_h + shadow_offset)],
        radius=card_radius, fill=shadow_color, outline=None
    )
    # main card
    draw.rounded_rectangle(
        [(card_x, y, card_x + card_w, y + card_h)],
        radius=card_radius, fill=card_fill, outline=card_outline, width=2
    )
    # left thumbnail background (placeholder background only; actual image/icon will be pasted on top)
    thumb_x = card_x + 24
    thumb_y = y + 24
    thumb_w = 210
    thumb_h = 210
    thumb_radius = 12
    thumb_bg = "#F1EEF2"  # muted placeholder
    draw.rounded_rectangle(
        [(thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h)],
        radius=thumb_radius, fill=thumb_bg, outline="#E8E3EA"
    )
    # small "badge" background near top-left of thumbnail (e.g., for free tag) - only background shape
    badge_w, badge_h = 84, 44
    badge_x = thumb_x + 8
    badge_y = thumb_y + 8
    badge_radius = 12
    badge_bg = "#E8F6EE"  # a gentle greenish tint for pill backgrounds
    draw.rounded_rectangle(
        [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
        radius=badge_radius, fill=badge_bg, outline=None
    )
    # vertical separator between thumbnail and text column (subtle)
    sep_x = thumb_x + thumb_w + 24
    draw.line([(sep_x, y + 20), (sep_x, y + card_h - 20)], fill="#F3EFF4", width=1)

    # card content area background block (light group background behind the title/metadata area)
    content_x = sep_x + 24
    content_y = y + 24
    content_w = card_x + card_w - content_x - 24
    content_h = thumb_h
    # subtle background behind text area to anchor layout (very light)
    draw.rectangle(
        [(content_x, content_y), (content_x + content_w, content_y + content_h)],
        fill=bg_color, outline=None
    )

    # right-side action icon background circle (placeholder only)
    action_cx = card_x + card_w - 72
    action_cy = y + card_h / 2
    action_r = 36
    action_bg = "#FFFFFF"
    draw.ellipse(
        [(action_cx - action_r, action_cy - action_r), (action_cx + action_r, action_cy + action_r)],
        fill=action_bg, outline="#ECE7EE", width=2
    )

    # subtle bottom divider for the card area
    draw.line([(card_x + 12, y + card_h + 6), (card_x + card_w - 12, y + card_h + 6)], fill="#F1EDF3", width=1)

# Global separators (between list groups)
for sep_y in [card_tops[0] - 24, card_tops[1] - 24, card_tops[2] - 24, card_tops[3] - 24]:
    draw.line([(48, sep_y), (1392, sep_y)], fill="#F4F1F6", width=1)

# Floating location pill/background is a detected element and will be pasted; draw a subtle shadow under where it will appear
# Detected location pill appears near y ~2651; draw only a shadow to anchor it
pill_shadow_top = 2620
pill_shadow_bottom = 2700
draw.ellipse([(480, pill_shadow_top + 6), (960, pill_shadow_bottom + 20)], fill="#F7F6F8")

# Bottom navigation bar background (safe to draw; icons will be pasted on top)
nav_h = 140
nav_top = 2820
nav_color = "#FFFFFF"
draw.rectangle([(0, nav_top), (1440, 2960)], fill=nav_color)
# top divider for nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#EDE8EE", width=2)

# small indicator above nav (a thin central notch area for active item background)
active_notch_w = 220
active_notch_h = 68
active_notch_x = (1440 - active_notch_w) / 2
active_notch_y = nav_top - 40
draw.rounded_rectangle(
    [(active_notch_x, active_notch_y), (active_notch_x + active_notch_w, active_notch_y + active_notch_h)],
    radius=36, fill="#FFFFFF", outline="#E9E3EA", width=2
)

# Final subtle vignette separators near page edges for visual polish (very light)
draw.line([(48, 420), (1392, 420)], fill="#F6F4F7", width=1)
draw.line([(48, 2400), (1392, 2400)], fill="#F6F4F7", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/00_icon_iORk.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["iORk"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/04_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/05_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/06_icon_The_DL.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["The_DL"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/07_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1678), _c7)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 763), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/10_icon_The_DL.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["The_DL"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/11_icon_8_8609Litutu.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["8_8609Litutu'"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/12_icon_The_DL.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["The_DL"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 58)
    canvas.paste(_c13, (183, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [183, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1284, 1159), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1539), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/16_icon_The_DL.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 2347), _c16)
except Exception:
    pass
layout["The_DL"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/17_icon_Favorite_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1140, 1159), _c17)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 56)
    canvas.paste(_c18, (247, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [247, 5, 303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/19_icon_E_PARTY.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (0, 2804), _c19)
except Exception:
    pass
layout["E_PARTY"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/20_icon_dtLaIct.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1678), _c20)
except Exception:
    pass
layout["dtLaIct"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 123)
    canvas.paste(_c21, (1284, 763), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/22_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 886), _c22)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/23_icon_New_York.png
try:
    _c23 = get_crop(23, 405, 117)
    canvas.paste(_c23, (518, 2651), _c23)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 47, 52)
    canvas.paste(_c24, (1321, 7), _c24)
except Exception:
    pass
layout["icon_24"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/25_icon_9.41.png
try:
    _c25 = get_crop(25, 93, 101)
    canvas.paste(_c25, (46, 120), _c25)
except Exception:
    pass
layout["9.41"] = [46, 120, 139, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 69, 59)
    canvas.paste(_c26, (1211, 4), _c26)
except Exception:
    pass
layout["icon_26"] = [1211, 4, 1280, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 62, 58)
    canvas.paste(_c27, (311, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [311, 5, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 56)
    canvas.paste(_c28, (383, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/29_icon_rJ_U_5I0.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (576, 2804), _c29)
except Exception:
    pass
layout["rJ'U'5I0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 42, 57)
    canvas.paste(_c30, (1272, 5), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 5, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/31_icon_Fireworks_July_Ath_Rooftop_Party.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Fireworks_July_Ath_Roofto"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/32_icon_Tickets.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (864, 2804), _c32)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/33_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/34_text_9.41.png
try:
    _c34 = get_crop(34, 89, 43)
    canvas.paste(_c34, (20, 15), _c34)
except Exception:
    pass
layout["9.41"] = [20, 15, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/35_text_More_events_you_II_love.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/36_text_onich.png
try:
    _c36 = get_crop(36, 199, 102)
    canvas.paste(_c36, (111, 2530), _c36)
except Exception:
    pass
layout["onich"] = [111, 2530, 310, 2632]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/37_text_Sat.png
try:
    _c37 = get_crop(37, 83, 48)
    canvas.paste(_c37, (388, 2554), _c37)
except Exception:
    pass
layout["Sat,_="] = [388, 2554, 471, 2602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/38_text_13_._11_30_PM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["13_._11:30_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/39_text_SHIINDIE.png
try:
    _c39 = get_crop(39, 195, 65)
    canvas.paste(_c39, (38, 2643), _c39)
except Exception:
    pass
layout["SHIINDIE"] = [38, 2643, 233, 2708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/40_text_E_PARTY.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (0, 2804), _c40)
except Exception:
    pass
layout["E_PARTY"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/41_text_rJ_U_5I0.png
try:
    _c41 = get_crop(41, 405, 117)
    canvas.paste(_c41, (518, 2651), _c41)
except Exception:
    pass
layout["rJ'U'5I0"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_01_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-3/42_clickable_More.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (1152, 2804), _c42)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
