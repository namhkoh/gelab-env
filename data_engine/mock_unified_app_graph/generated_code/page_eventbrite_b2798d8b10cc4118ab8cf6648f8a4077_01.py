# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_01
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3.png
# step_index: 1/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (249, 249, 251)          # overall very light off-white background
status_bar_color = (230, 230, 232)  # light grey for status bar area
header_bg = (255, 255, 255)         # white for header/toolbars
divider_color = (235, 231, 241)     # soft lavender divider
card_alt_color = (247, 244, 252)    # subtle card background (very light lavender)
card_outline = (235, 235, 236)      # faint outline for cards
nav_bg = (255, 255, 255)            # bottom nav background (white)
nav_top_border = (230, 230, 232)    # top border for nav

width, height = canvas.size

# Fill overall background
draw.rectangle((0, 0, width, height), fill=bg_color)

# Status bar area (~50px tall)
status_h = 90
draw.rectangle((0, 0, width, status_h), fill=status_bar_color)

# Header / toolbar background (area containing search field)
header_top = status_h
header_bottom = 220
draw.rectangle((0, header_top, width, header_bottom), fill=header_bg)

# Light divider below header
draw.line((48, header_bottom, width-48, header_bottom), fill=divider_color, width=2)

# Section: "More events you'll love" heading area has white bg, keep it clean.
# Add a faint horizontal divider under the heading area (above the list)
heading_div_y = 450
draw.line((48, heading_div_y, width-48, heading_div_y), fill=divider_color, width=1)

# Draw alternating subtle card backgrounds for each event block.
# Blocks detected in the UI at y positions (from detection): 490, 886, 1282, 1678, 2074, 2470
block_positions = [490, 886, 1282, 1678, 2074, 2470]
block_width = 1344
left = 48
right = left + block_width
block_height = 396
radius = 14

for i, y in enumerate(block_positions):
    top = y
    bottom = y + block_height
    # Skip drawing inside floating "Find / Online events" pill region to avoid duplicating detected element:
    # pill bbox: (427,2651) size 586x117 -> region x=427..1013, y=2651..2768
    pill_x1, pill_y1, pill_x2, pill_y2 = 427, 2651, 1013, 2768
    # Choose background color for alternating cards
    fill_color = card_alt_color if (i % 2 == 0) else header_bg
    # Draw subtle shadow underneath card (slightly offset, low-opacity effect simulated with faint line)
    shadow_y_offset = 6
    # shadow rectangle (very faint)
    draw.rounded_rectangle(
        (left + 2, top + shadow_y_offset, right + 2, bottom + shadow_y_offset),
        radius=radius,
        fill=(245, 244, 246)
    )
    # Main card rounded rect
    card_bbox = (left, top, right, bottom)
    # If the card overlaps the floating pill region, clip the card drawing to avoid drawing over the pill area.
    # We'll draw three pieces: left part (if any), top part above pill, and right part (if any) excluding pill bbox.
    # Simple approach: if vertical overlap with pill, draw card in two halves split horizontally outside the pill bbox.
    if not (top < pill_y2 and bottom > pill_y1):
        # no vertical overlap -> safe to draw whole card
        draw.rounded_rectangle(card_bbox, radius=radius, fill=fill_color, outline=card_outline, width=1)
    else:
        # draw top segment above pill
        if top < pill_y1:
            seg_top = top
            seg_bottom = min(bottom, pill_y1 - 6)
            if seg_bottom > seg_top:
                draw.rounded_rectangle((left, seg_top, right, seg_bottom), radius=radius, fill=fill_color, outline=card_outline, width=1)
        # draw bottom segment below pill
        if bottom > pill_y2:
            seg_top = max(top, pill_y2 + 6)
            seg_bottom = bottom
            if seg_bottom > seg_top:
                draw.rounded_rectangle((left, seg_top, right, seg_bottom), radius=radius, fill=fill_color, outline=card_outline, width=1)
        # draw left strip (in case pill overlaps centrally) for the vertical overlap area
        overlap_top = max(top, pill_y1)
        overlap_bottom = min(bottom, pill_y2)
        if overlap_bottom > overlap_top:
            # left strip
            left_strip_right = min(pill_x1 - 8, right)
            if left_strip_right > left:
                draw.rectangle((left, overlap_top, left_strip_right, overlap_bottom), fill=fill_color, outline=None)
            # right strip
            right_strip_left = max(pill_x2 + 8, left)
            if right_strip_left < right:
                draw.rectangle((right_strip_left, overlap_top, right, overlap_bottom), fill=fill_color, outline=None)
        # Finally draw a faint outline around the whole card region (excluding pill box); draw thin lines at top/bottom edges
        draw.line((left, top, right, top), fill=card_outline, width=1)
        draw.line((left, bottom, right, bottom), fill=card_outline, width=1)

    # Add a subtle horizontal separator at the bottom edge of each block
    draw.line((left + 8, bottom + 8, right - 8, bottom + 8), fill=divider_color, width=1)

# Add thin separators between each card (full width minus margins)
for y in [pos + block_height for pos in block_positions]:
    sep_y = y + 4
    if sep_y < height - 160:  # don't draw over the bottom nav
        draw.line((48, sep_y, width - 48, sep_y), fill=divider_color, width=1)

# Content area background for the very bottom section (above nav) - keep clean white
bottom_section_top = 2680
draw.rectangle((0, bottom_section_top, width, 2804), fill=header_bg)

# Bottom navigation bar background and top border
nav_top = 2804
draw.rectangle((0, nav_top, width, height), fill=nav_bg)
draw.line((0, nav_top, width, nav_top), fill=nav_top_border, width=2)

# Add subtle indicator for safe area and separation for nav icons (no icons drawn)
# center small horizontal guideline (very faint) where icons will sit
icon_guide_y = nav_top + 82
draw.line((72, icon_guide_y, width - 72, icon_guide_y), fill=(250,250,251), width=1)

# Final thin left and right margins to frame content area
draw.line((48, header_bottom, 48, height - 160), fill=(252,252,253), width=1)
draw.line((width - 48, header_bottom, width - 48, height - 160), fill=(252,252,253), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/00_icon_ASW-NJ_Grief_Certificate_Progral.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["ASW-NJ_Grief_Certificate_"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/01_icon_dolescents.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1282), _c1)
except Exception:
    pass
layout["dolescents"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/03_icon_Therapeutic_Practice.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1678), _c3)
except Exception:
    pass
layout["Therapeutic_Practice"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 123)
    canvas.paste(_c4, (1140, 1555), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1284, 1555), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/06_icon_Or.png
try:
    _c6 = get_crop(6, 288, 156)
    canvas.paste(_c6, (288, 2804), _c6)
except Exception:
    pass
layout["Or"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1951), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/08_icon_Srief_and_Loss.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 2347), _c8)
except Exception:
    pass
layout["Srief_and_Loss"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 1143), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/10_icon_Online.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1678), _c10)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 747), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 2347), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1143), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/15_icon_Home.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (0, 2804), _c15)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/16_icon_9.18.png
try:
    _c16 = get_crop(16, 53, 58)
    canvas.paste(_c16, (183, 3), _c16)
except Exception:
    pass
layout["9.18"] = [183, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/17_icon_Favorite_button.png
try:
    _c17 = get_crop(17, 144, 139)
    canvas.paste(_c17, (1140, 747), _c17)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 58)
    canvas.paste(_c18, (247, 4), _c18)
except Exception:
    pass
layout["icon_18"] = [247, 4, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 48, 53)
    canvas.paste(_c19, (1321, 7), _c19)
except Exception:
    pass
layout["icon_19"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/20_icon_9.18.png
try:
    _c20 = get_crop(20, 97, 99)
    canvas.paste(_c20, (43, 121), _c20)
except Exception:
    pass
layout["9.18"] = [43, 121, 140, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/21_icon_8_15_AM_GMT.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1282), _c21)
except Exception:
    pass
layout["8:15_AM_GMT"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 84, 59)
    canvas.paste(_c22, (1211, 4), _c22)
except Exception:
    pass
layout["icon_22"] = [1211, 4, 1295, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/23_icon_8_1234_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 490), _c23)
except Exception:
    pass
layout["8_1234_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/24_icon_Understanding_Grief_and_Loss.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 886), _c24)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/25_icon_Tr.png
try:
    _c25 = get_crop(25, 56, 55)
    canvas.paste(_c25, (389, 2641), _c25)
except Exception:
    pass
layout["Tr"] = [389, 2641, 445, 2696]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 47, 56)
    canvas.paste(_c26, (384, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [384, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/27_icon_A_Certificate_Program_Bundle.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 2074), _c27)
except Exception:
    pass
layout["A_Certificate_Program_(Bu"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 62, 61)
    canvas.paste(_c28, (311, 4), _c28)
except Exception:
    pass
layout["icon_28"] = [311, 4, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/29_icon_Online_events.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (576, 2804), _c29)
except Exception:
    pass
layout["Online_events"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/30_icon_Online.png
try:
    _c30 = get_crop(30, 111, 53)
    canvas.paste(_c30, (391, 703), _c30)
except Exception:
    pass
layout["Online"] = [391, 703, 502, 756]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/31_icon_Free.png
try:
    _c31 = get_crop(31, 119, 74)
    canvas.paste(_c31, (252, 558), _c31)
except Exception:
    pass
layout["Free"] = [252, 558, 371, 632]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/32_icon_Grief_and.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 490), _c32)
except Exception:
    pass
layout["Grief_and"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/33_icon_Understanding_Grief_and_Loss.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 490), _c33)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 42, 57)
    canvas.paste(_c34, (1272, 5), _c34)
except Exception:
    pass
layout["icon_34"] = [1272, 5, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/35_icon_Online.png
try:
    _c35 = get_crop(35, 112, 50)
    canvas.paste(_c35, (390, 1528), _c35)
except Exception:
    pass
layout["Online"] = [390, 1528, 502, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/36_icon_Online_events.png
try:
    _c36 = get_crop(36, 586, 117)
    canvas.paste(_c36, (427, 2651), _c36)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/37_icon_Tickets.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (864, 2804), _c37)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/38_icon_10_CEU_NASW-NJ_Navigating_Grief_and_Loss.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 2074), _c38)
except Exception:
    pass
layout["10_CEU_NASW-NJ_Navigating"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/39_icon_Online.png
try:
    _c39 = get_crop(39, 111, 53)
    canvas.paste(_c39, (390, 1923), _c39)
except Exception:
    pass
layout["Online"] = [390, 1923, 501, 1976]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/40_icon_Understanding_Grief_and_Loss_Foundation_.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 1678), _c40)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/41_text_9.18.png
try:
    _c41 = get_crop(41, 94, 41)
    canvas.paste(_c41, (20, 17), _c41)
except Exception:
    pass
layout["9.18"] = [20, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/42_text_More_events_you_II_love.png
try:
    _c42 = get_crop(42, 1344, 396)
    canvas.paste(_c42, (48, 490), _c42)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/43_text_Sat_May_4_._3.30_PM_GMT.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["Sat,_May_4_._3.30_PM_GMT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/44_text_Or.png
try:
    _c44 = get_crop(44, 50, 31)
    canvas.paste(_c44, (393, 2722), _c44)
except Exception:
    pass
layout["Or"] = [393, 2722, 443, 2753]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_01_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
