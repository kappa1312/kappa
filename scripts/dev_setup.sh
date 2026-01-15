#!/usr/bin/env bash
# Kappa OS Development Setup Script
# Usage: ./scripts/dev_setup.sh

set -e

echo "======================================"
echo "  Kappa OS Development Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}Error: Python 3.11+ is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION OK${NC}"

# Check Poetry
echo ""
echo "Checking Poetry..."
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}Poetry not found. Installing...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
POETRY_VERSION=$(poetry --version 2>&1 | awk '{print $3}')
echo -e "${GREEN}Poetry $POETRY_VERSION OK${NC}"

# Check Node.js (required for Claude Code CLI)
echo ""
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Warning: Node.js not found. Claude Agent SDK requires Node.js 18+${NC}"
    echo "Install from: https://nodejs.org/"
else
    NODE_VERSION=$(node --version | sed 's/v//')
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)
    if [ "$NODE_MAJOR" -lt 18 ]; then
        echo -e "${YELLOW}Warning: Node.js 18+ recommended (found $NODE_VERSION)${NC}"
    else
        echo -e "${GREEN}Node.js $NODE_VERSION OK${NC}"
    fi
fi

# Check PostgreSQL
echo ""
echo "Checking PostgreSQL..."
if ! command -v psql &> /dev/null; then
    echo -e "${YELLOW}Warning: PostgreSQL client not found${NC}"
    echo "Install PostgreSQL or ensure psql is in PATH"
else
    PSQL_VERSION=$(psql --version | awk '{print $3}')
    echo -e "${GREEN}PostgreSQL client $PSQL_VERSION OK${NC}"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
poetry install

# Create .env if not exists
echo ""
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo -e "${YELLOW}Created .env - please update with your credentials${NC}"
else
    echo -e "${GREEN}.env already exists${NC}"
fi

# Create logs directory
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p workspace
echo -e "${GREEN}Directories created${NC}"

# Database setup instructions
echo ""
echo "======================================"
echo "  Database Setup"
echo "======================================"
echo ""
echo "To set up the database, run:"
echo ""
echo "  1. Create database:"
echo "     createdb kappa_db"
echo ""
echo "  2. Run schema setup:"
echo "     psql -d kappa_db -f scripts/setup_db.sql"
echo ""
echo "  3. Update DATABASE_URL in .env"
echo ""

# Verify installation
echo ""
echo "======================================"
echo "  Verification"
echo "======================================"
echo ""
echo "Running verification checks..."

# Check imports
poetry run python -c "from src.core.config import Settings; print('Config import OK')" 2>/dev/null && \
    echo -e "${GREEN}Config module OK${NC}" || \
    echo -e "${RED}Config module failed${NC}"

poetry run python -c "from src.cli.main import app; print('CLI import OK')" 2>/dev/null && \
    echo -e "${GREEN}CLI module OK${NC}" || \
    echo -e "${RED}CLI module failed${NC}"

# Final message
echo ""
echo "======================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Update .env with your ANTHROPIC_API_KEY"
echo "  2. Set up PostgreSQL database"
echo "  3. Run: poetry run kappa --help"
echo ""
