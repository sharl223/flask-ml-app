#!/usr/bin/env python3
"""
AI Playground - Favicon Generator
Generate favicon files for the web application
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_favicon():
    """Create favicon files for the application"""
    
    try:
        # Create static directory if it doesn't exist
        static_dir = 'static'
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            print(f"Created directory: {static_dir}")
        
        # Create a simple AI-themed favicon
        def create_icon(size, filename):
            try:
                # Create a new image with a dark blue background
                img = Image.new('RGBA', (size, size), (25, 118, 210, 255))
                draw = ImageDraw.Draw(img)
                
                # Draw a simple AI symbol (circuit-like pattern)
                margin = size // 4
                center = size // 2
                
                # Draw outer circle
                draw.ellipse([margin, margin, size - margin, size - margin], 
                            outline=(255, 255, 255, 255), width=max(1, size // 32))
                
                # Draw inner elements
                if size >= 32:
                    # Draw small circles at corners
                    small_margin = size // 8
                    small_radius = size // 16
                    positions = [
                        (small_margin, small_margin),
                        (size - small_margin, small_margin),
                        (small_margin, size - small_margin),
                        (size - small_margin, size - small_margin)
                    ]
                    
                    for pos in positions:
                        draw.ellipse([pos[0] - small_radius, pos[1] - small_radius,
                                    pos[0] + small_radius, pos[1] + small_radius],
                                   fill=(255, 255, 255, 255))
                
                # Save the image
                filepath = os.path.join(static_dir, filename)
                img.save(filepath, 'PNG')
                print(f"Created: {filename} ({os.path.getsize(filepath)} bytes)")
                return True
            except Exception as e:
                print(f"Error creating {filename}: {e}")
                return False
        
        # Create different sizes
        sizes = [
            (16, 'favicon-16x16.png'),
            (32, 'favicon-32x32.png'),
            (192, 'android-chrome-192x192.png'),
            (512, 'android-chrome-512x512.png'),
            (180, 'apple-touch-icon.png')
        ]
        
        success_count = 0
        for size, filename in sizes:
            if create_icon(size, filename):
                success_count += 1
        
        # Create ICO file (multi-size)
        try:
            ico_sizes = [16, 32, 48]
            ico_images = []
            
            for size in ico_sizes:
                img = Image.new('RGBA', (size, size), (25, 118, 210, 255))
                draw = ImageDraw.Draw(img)
                
                margin = size // 4
                draw.ellipse([margin, margin, size - margin, size - margin], 
                            outline=(255, 255, 255, 255), width=max(1, size // 32))
                
                # Convert to RGB for ICO
                rgb_img = Image.new('RGB', img.size, (25, 118, 210))
                rgb_img.paste(img, mask=img.split()[-1])
                ico_images.append(rgb_img)
            
            ico_path = os.path.join(static_dir, 'favicon.ico')
            ico_images[0].save(ico_path, format='ICO', sizes=[(size, size) for size in ico_sizes])
            print(f"Created: favicon.ico ({os.path.getsize(ico_path)} bytes)")
            success_count += 1
        except Exception as e:
            print(f"Error creating favicon.ico: {e}")
        
        # Create web manifest
        try:
            manifest_content = '''{
  "name": "AI Playground",
  "short_name": "AI Playground",
  "description": "機械学習とデータ分析のための総合プラットフォーム / Comprehensive platform for machine learning and data analysis",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1976d2",
  "theme_color": "#1976d2",
  "icons": [
    {
      "src": "/static/android-chrome-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/android-chrome-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}'''
            
            manifest_path = os.path.join(static_dir, 'site.webmanifest')
            with open(manifest_path, 'w', encoding='utf-8') as f:
                f.write(manifest_content)
            print(f"Created: site.webmanifest ({os.path.getsize(manifest_path)} bytes)")
            success_count += 1
        except Exception as e:
            print(f"Error creating site.webmanifest: {e}")
        
        print(f"Favicon generation completed! {success_count}/{len(sizes) + 2} files created successfully.")
        
    except Exception as e:
        print(f"Fatal error in favicon generation: {e}")

if __name__ == '__main__':
    create_favicon() 