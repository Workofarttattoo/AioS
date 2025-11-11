/**
 * Red Team Tools Suite - Main JavaScript
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)
 */

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add scroll effect to navbar
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.5)';
    } else {
        navbar.style.boxShadow = 'none';
    }
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all cards
document.querySelectorAll('.tool-card, .algo-category, .integration-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    observer.observe(card);
});

// Copy code blocks to clipboard
document.querySelectorAll('.code-block').forEach(block => {
    block.addEventListener('click', () => {
        const code = block.textContent;
        navigator.clipboard.writeText(code).then(() => {
            const tooltip = document.createElement('div');
            tooltip.textContent = 'Copied!';
            tooltip.style.cssText = `
                position: absolute;
                background: var(--primary);
                color: var(--dark-bg);
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 600;
                pointer-events: none;
                animation: fadeOut 2s forwards;
            `;
            block.style.position = 'relative';
            block.appendChild(tooltip);
            setTimeout(() => tooltip.remove(), 2000);
        });
    });
});

// Mobile menu toggle (if needed)
const createMobileMenu = () => {
    const nav = document.querySelector('.nav-links');
    const menuBtn = document.createElement('button');
    menuBtn.classList.add('mobile-menu-btn');
    menuBtn.innerHTML = '☰';
    menuBtn.style.cssText = `
        display: none;
        font-size: 1.5rem;
        background: none;
        border: none;
        color: var(--primary);
        cursor: pointer;
    `;

    if (window.innerWidth <= 768) {
        menuBtn.style.display = 'block';
        document.querySelector('.navbar .container').appendChild(menuBtn);

        menuBtn.addEventListener('click', () => {
            nav.style.display = nav.style.display === 'flex' ? 'none' : 'flex';
        });
    }
};

window.addEventListener('resize', createMobileMenu);
createMobileMenu();

console.log('%c🛡️ Red Team Tools Suite', 'font-size: 20px; font-weight: bold; color: #00ff88;');
console.log('%cBuilt by Joshua Hendricks Cole | Corporation of Light', 'color: #8b92a8;');
console.log('%cOpen source, free forever. Inspect away!', 'color: #0088ff;');
