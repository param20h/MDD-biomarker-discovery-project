// Smooth scrolling for navigation links
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

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all cards and sections
document.querySelectorAll('.overview-card, .feature-box, .viz-card, .qa-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Add active state to navigation on scroll
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-menu a');

window.addEventListener('scroll', () => {
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= (sectionTop - 200)) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').slice(1) === current) {
            link.classList.add('active');
        }
    });
});

// Counter animation for stat numbers
function animateCounter(element, target, duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16);
    
    const timer = setInterval(() => {
        start += increment;
        if (start >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(start);
        }
    }, 16);
}

// Trigger counter animation when hero stats are visible
const heroStats = document.querySelector('.hero-stats');
if (heroStats) {
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                document.querySelectorAll('.stat-number').forEach(stat => {
                    const target = parseFloat(stat.textContent);
                    if (!isNaN(target) && target > 1) {
                        stat.textContent = '0';
                        animateCounter(stat, target);
                    }
                });
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    statsObserver.observe(heroStats);
}

// Add hover effect to tech badges
document.querySelectorAll('.tech-badge').forEach(badge => {
    badge.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-5px) scale(1.05)';
    });
    
    badge.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Mobile menu toggle (for future mobile navigation)
const createMobileMenu = () => {
    const nav = document.querySelector('.navbar');
    const menu = document.querySelector('.nav-menu');
    
    if (window.innerWidth <= 768) {
        if (!document.querySelector('.mobile-menu-toggle')) {
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'mobile-menu-toggle';
            toggleBtn.innerHTML = '<i class="fas fa-bars"></i>';
            toggleBtn.style.cssText = 'background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer; display: block;';
            
            toggleBtn.addEventListener('click', () => {
                menu.classList.toggle('active');
                menu.style.display = menu.style.display === 'flex' ? 'none' : 'flex';
                menu.style.flexDirection = 'column';
                menu.style.position = 'absolute';
                menu.style.top = '100%';
                menu.style.left = '0';
                menu.style.right = '0';
                menu.style.background = 'var(--dark-bg)';
                menu.style.padding = '1rem';
            });
            
            nav.querySelector('.container').appendChild(toggleBtn);
        }
    }
};

window.addEventListener('resize', createMobileMenu);
createMobileMenu();

// Add clipboard copy functionality for code snippets
document.querySelectorAll('pre code').forEach(block => {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'Copy';
    button.style.cssText = 'position: absolute; top: 10px; right: 10px; padding: 5px 10px; background: var(--primary-color); color: white; border: none; border-radius: 5px; cursor: pointer;';
    
    const pre = block.parentElement;
    pre.style.position = 'relative';
    pre.appendChild(button);
    
    button.addEventListener('click', () => {
        navigator.clipboard.writeText(block.textContent);
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    });
});

// Parallax effect for hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    if (hero && scrolled < hero.offsetHeight) {
        hero.style.transform = `translateY(${scrolled * 0.5}px)`;
        hero.style.opacity = 1 - (scrolled / hero.offsetHeight);
    }
});

// Load external visualizations (if available)
const loadVisualizations = async () => {
    const vizPlaceholders = document.querySelectorAll('.viz-placeholder');
    
    // In a real implementation, you would fetch actual chart images here
    vizPlaceholders.forEach((placeholder, index) => {
        const mockCharts = [
            'PCA scree plot visualization',
            't-SNE 2D cluster visualization',
            'Heatmap of cluster vs PHQ-8',
            'Silhouette score analysis'
        ];
        
        // Placeholder for future chart integration
        placeholder.innerHTML = `
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 3rem; 
                        border-radius: 10px; 
                        color: white;
                        opacity: 0.7;">
                <i class="fas fa-chart-bar" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                <p>${mockCharts[index]}</p>
                <small>View in Jupyter Notebook for actual plots</small>
            </div>
        `;
    });
};

loadVisualizations();

console.log('🧠 Depression Biomarker Discovery Project - Interactive Dashboard Loaded');
console.log('📊 Results: 2 subtypes discovered, p=0.0112 (significant!)');
